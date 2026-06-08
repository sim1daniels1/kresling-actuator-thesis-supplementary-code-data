// abp_fracture_return_to_zero_with_closed_loop_holds.ino
#include <AccelStepper.h>
#include <math.h>

// ===================== USER SETTINGS =====================

// ---- PRE/POST RAMP (reach -> hold -> step) ----
const float  RAMP_BASE_GAUGE_KPA   = 0.0f;
const float  RAMP_MAX_GAUGE_KPA    = 100.0f;
const float  RAMP_STEP_GAUGE_KPA   = 10.0f;

const unsigned long HOLD_AT_TARGET_MS = 3000UL;
const float ENTER_HOLD_KPA = 2.0f;
const float STAY_HOLD_KPA  = 3.0f;

// ---- FRACTURE PROGRAM (cycles with increasing P_high) ----
const float P_MIN_GAUGE_KPA        = 0.0f;
const float P_MAX_FINAL_GAUGE_KPA  = 300.0f;
const float STEP_GAUGE_KPA         = 50.0f;
const int   CYCLES_PER_STEP        = 10;
const unsigned long HOLD_BETWEEN_STEPS_MS = 1500UL;

// ---- Per-cycle dwells ----
const unsigned long HOLD_AT_HIGH_MS = 3000UL; // dwell at top of each cycle
const unsigned long HOLD_AT_LOW_MS  = 3000UL; // dwell at bottom of each cycle

// ---- Control deadband for switching (general) ----
float margin_kPa = 2.0f;

// ---- HOLD control (gentle closed-loop) ----
const float HOLD_BAND_KPA = 2.0f;    // maintain within ± this during holds
float SPEED_HOLD_UP   = 250.0f;      // gentle nudges
float SPEED_HOLD_DOWN = -250.0f;

// ---- Speeds (microsteps/s) ----
float SPEED_RAMP_UP   = 300.0f;
float SPEED_RAMP_DOWN = -300.0f;

float SPEED_CYCLE_UP   = 1500.0f;
float SPEED_CYCLE_DOWN = -1500.0f;

// ---- Direction invert ----
int DIR_SIGN = -1;

// ---- Safety ----
const float P_HARD_MAX_GAUGE_KPA = 320.0f;

// ---- Return-to-zero before starting test ----
const float RETURN_ZERO_TARGET_KPA = 0.0f;
const float RETURN_ZERO_TOL_KPA    = 2.0f;
const float RETURN_ZERO_ONLY_IF_ABOVE_KPA = 3.0f;

// ==========================================================

// ---- Pins ----
const int STEP_PIN  = 2;
const int DIR_PIN   = 5;
const int PRESS_PIN = A0;

// ---- Stepper ----
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ---- Pressure averaging ----
const int NAVG = 3;

// ---- State machine ----
enum State {
  IDLE,
  RETURN_TO_ZERO,
  RAMP_PRE,
  GO_UP,
  HOLD_HIGH,
  GO_DOWN,
  HOLD_LOW,
  STEP_DWELL,
  RAMP_POST,
  FINAL_RETURN_TO_ZERO,
  DONE
};

State state = IDLE;
bool testRunning = false;

// ---- Ramp vars ----
float rampTargetGauge = RAMP_BASE_GAUGE_KPA;
bool rampInHold = false;
unsigned long rampHoldStartMs = 0;

// ---- Fracture vars ----
float P_high_gauge = P_MIN_GAUGE_KPA + STEP_GAUGE_KPA;
int cyclesThisStep = 0;
unsigned long dwellStartMs = 0;
bool highReached = false;

// ---- Per-cycle hold timer ----
unsigned long cycleHoldStartMs = 0;

// ============================================================================
// ABP 060PGAA5 transfer function: 10–90% Vs for 0..60 psi gauge (0..413.7 kPa_g)
// ============================================================================
const float P_GAUGE_FULLSCALE_KPA = 413.7f;

static inline float adcToGauge_kPa(int adc) {
  float r = adc / 1023.0f;
  float Pg = (r - 0.10f) / 0.80f * P_GAUGE_FULLSCALE_KPA;
  if (Pg < -10.0f) Pg = -10.0f; // allow small negatives due to offset/noise
  return Pg;
}

float readGauge_kPa() {
  static long acc = 0;
  static int  cnt = 0;

  int val = analogRead(PRESS_PIN);
  acc += val;
  cnt++;

  if (cnt < NAVG) return NAN;

  int avg = acc / cnt;
  acc = 0;
  cnt = 0;

  return adcToGauge_kPa(avg);
}

// ============================================================================
// Bang-bang to target (general, uses margin_kPa)
// ============================================================================
void driveToTarget(float P_g, float target_g, float speedUp, float speedDown) {
  if (P_g < target_g - margin_kPa) {
    stepper.setSpeed(DIR_SIGN * speedUp);
  } else if (P_g > target_g + margin_kPa) {
    stepper.setSpeed(DIR_SIGN * speedDown);
  } else {
    stepper.setSpeed(0.0f);
  }
}

// ============================================================================
// Closed-loop HOLD control (gentle, uses HOLD_BAND_KPA)
// ============================================================================
void holdToTarget(float P_g, float target_g) {
  if (P_g < target_g - HOLD_BAND_KPA) {
    stepper.setSpeed(DIR_SIGN * SPEED_HOLD_UP);
  } else if (P_g > target_g + HOLD_BAND_KPA) {
    stepper.setSpeed(DIR_SIGN * SPEED_HOLD_DOWN);
  } else {
    stepper.setSpeed(0.0f);
  }
}

// ============================================================================
// Return-to-zero (only bleed down, don't pump up)
// ============================================================================
bool runReturnToZero(float P_g) {
  if (P_g > RETURN_ZERO_TARGET_KPA + RETURN_ZERO_TOL_KPA) {
    stepper.setSpeed(DIR_SIGN * SPEED_RAMP_DOWN);
    return false;
  }
  stepper.setSpeed(0.0f);
  return true;
}

// ============================================================================
// Ramp reach/hold/step
// ============================================================================
void resetRamp() {
  rampTargetGauge = RAMP_BASE_GAUGE_KPA;
  rampInHold = false;
  rampHoldStartMs = 0;
}

bool runRampReachHold(float P_g, unsigned long nowMs, const __FlashStringHelper* label) {
  driveToTarget(P_g, rampTargetGauge, SPEED_RAMP_UP, SPEED_RAMP_DOWN);

  bool withinEnter = (fabs(P_g - rampTargetGauge) <= ENTER_HOLD_KPA);
  bool withinStay  = (fabs(P_g - rampTargetGauge) <= STAY_HOLD_KPA);

  if (!rampInHold) {
    if (withinEnter) {
      rampInHold = true;
      rampHoldStartMs = nowMs;
      Serial.print(label); Serial.print(F(" reached "));
      Serial.print(rampTargetGauge, 1);
      Serial.println(F(" kPa_g -> HOLD"));
    }
  } else {
    if (!withinStay) {
      rampInHold = false;
      Serial.print(label); Serial.print(F(" drifted from "));
      Serial.print(rampTargetGauge, 1);
      Serial.println(F(" kPa_g -> re-acquire"));
    } else {
      if (nowMs - rampHoldStartMs >= HOLD_AT_TARGET_MS) {
        rampInHold = false;

        if (rampTargetGauge < RAMP_MAX_GAUGE_KPA) {
          rampTargetGauge += RAMP_STEP_GAUGE_KPA;
          if (rampTargetGauge > RAMP_MAX_GAUGE_KPA) rampTargetGauge = RAMP_MAX_GAUGE_KPA;

          Serial.print(label); Serial.print(F(" step -> "));
          Serial.print(rampTargetGauge, 1);
          Serial.println(F(" kPa_g"));
        } else {
          return true;
        }
      }
    }
  }
  return false;
}

// ============================================================================
// Start / Stop
// ============================================================================
void startTest() {
  stepper.setSpeed(0.0f);
  stepper.setCurrentPosition(0);

  resetRamp();

  P_high_gauge = P_MIN_GAUGE_KPA + STEP_GAUGE_KPA;
  if (P_high_gauge > P_MAX_FINAL_GAUGE_KPA) P_high_gauge = P_MAX_FINAL_GAUGE_KPA;

  cyclesThisStep = 0;
  highReached = false;

  // initial pressure estimate for deciding return-to-zero
  float sum_g = 0.0f;
  for (int i = 0; i < 30; i++) {
    sum_g += adcToGauge_kPa(analogRead(PRESS_PIN));
    delay(5);
  }
  float Pg_init = sum_g / 30.0f;

  testRunning = true;

  Serial.print(F("Test started. Pg_init="));
  Serial.print(Pg_init, 2);
  Serial.println(F(" kPa_g"));

  if (Pg_init > RETURN_ZERO_ONLY_IF_ABOVE_KPA) {
    state = RETURN_TO_ZERO;
    Serial.println(F("Initial Pg above threshold -> RETURN_TO_ZERO before RAMP_PRE."));
  } else {
    state = RAMP_PRE;
    Serial.println(F("Initial Pg near zero -> starting RAMP_PRE."));
  }
}

void stopTest(const __FlashStringHelper* msg) {
  testRunning = false;
  stepper.setSpeed(0.0f);
  state = IDLE;
  Serial.println(msg);
}

// ============================================================================
// Setup
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  float maxAbs = fabs(SPEED_RAMP_UP);
  if (fabs(SPEED_RAMP_DOWN) > maxAbs) maxAbs = fabs(SPEED_RAMP_DOWN);
  if (fabs(SPEED_CYCLE_UP) > maxAbs)  maxAbs = fabs(SPEED_CYCLE_UP);
  if (fabs(SPEED_CYCLE_DOWN) > maxAbs)maxAbs = fabs(SPEED_CYCLE_DOWN);
  if (fabs(SPEED_HOLD_UP) > maxAbs)   maxAbs = fabs(SPEED_HOLD_UP);
  if (fabs(SPEED_HOLD_DOWN) > maxAbs) maxAbs = fabs(SPEED_HOLD_DOWN);

  stepper.setMaxSpeed(maxAbs + 200.0f);
  stepper.setAcceleration(1200.0f);
  stepper.setSpeed(0.0f);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN,  OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN,  LOW);

  Serial.println(F("Rig ready (ABP gauge only)."));
  Serial.println(F("Commands: 's' start, 'x' stop, 'i' invert dir"));
}

// ============================================================================
// Loop
// ============================================================================
void loop() {
  static unsigned long lastPrintMs = 0;
  unsigned long nowMs = millis();

  // Serial commands
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'i' || c == 'I') {
      DIR_SIGN = -DIR_SIGN;
      Serial.print(F("Direction inverted. DIR_SIGN="));
      Serial.println(DIR_SIGN);
    } else if ((c == 's' || c == 'S') && !testRunning) {
      startTest();
    } else if (c == 'x' || c == 'X') {
      stopTest(F("Test stopped."));
    }
  }

  // Pressure read
  float P_g = readGauge_kPa();
  if (isnan(P_g)) {
    stepper.runSpeed();
    return;
  }

  // Safety
  if (testRunning && P_g >= P_HARD_MAX_GAUGE_KPA) {
    stopTest(F("SAFETY STOP: gauge pressure exceeded hard limit."));
  }

  // State machine
  if (testRunning) {
    switch (state) {

      case RETURN_TO_ZERO: {
        bool done = runReturnToZero(P_g);
        if (done) {
          state = RAMP_PRE;
          Serial.println(F("RETURN_TO_ZERO complete -> starting RAMP_PRE."));
        }
      } break;

      case RAMP_PRE: {
        bool done = runRampReachHold(P_g, nowMs, F("RAMP_PRE"));
        if (done) {
          state = GO_UP;
          stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_UP);
          Serial.println(F("RAMP_PRE complete -> starting fracture cycling."));
          Serial.print(F("Step start: Phigh="));
          Serial.print(P_high_gauge, 1);
          Serial.println(F(" kPa_g"));
        }
      } break;

      case GO_UP: {
        if (P_g < P_high_gauge - margin_kPa) stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_UP);
        else stepper.setSpeed(0.0f);

        if (P_g >= P_high_gauge - margin_kPa) {
          highReached = true;
          state = HOLD_HIGH;
          cycleHoldStartMs = nowMs;
          Serial.print(F("Reached Phigh="));
          Serial.print(P_high_gauge, 1);
          Serial.println(F(" kPa_g -> HOLD_HIGH"));
        }
      } break;

      case HOLD_HIGH: {
        // Actively maintain Phigh during dwell
        holdToTarget(P_g, P_high_gauge);

        if (nowMs - cycleHoldStartMs >= HOLD_AT_HIGH_MS) {
          state = GO_DOWN;
          stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_DOWN);
          Serial.println(F("HOLD_HIGH done -> DOWN"));
        }
      } break;

      case GO_DOWN: {
        if (P_g > P_MIN_GAUGE_KPA + margin_kPa) stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_DOWN);
        else stepper.setSpeed(0.0f);

        if (P_g <= P_MIN_GAUGE_KPA + margin_kPa) {
          state = HOLD_LOW;
          cycleHoldStartMs = nowMs;
          Serial.println(F("Reached Pmin -> HOLD_LOW"));
        }
      } break;

      case HOLD_LOW: {
        // Actively maintain Pmin during dwell
        holdToTarget(P_g, P_MIN_GAUGE_KPA);

        if (nowMs - cycleHoldStartMs >= HOLD_AT_LOW_MS) {

          if (highReached) {
            cyclesThisStep++;
            highReached = false;

            Serial.print(F("Cycle "));
            Serial.print(cyclesThisStep);
            Serial.print(F("/"));
            Serial.print(CYCLES_PER_STEP);
            Serial.print(F(" complete at Phigh="));
            Serial.print(P_high_gauge, 1);
            Serial.println(F(" kPa_g"));
          }

          if (cyclesThisStep >= CYCLES_PER_STEP) {
            state = STEP_DWELL;
            dwellStartMs = nowMs;
            stepper.setSpeed(0.0f);
            Serial.println(F("Step complete -> dwell at Pmin"));
          } else {
            state = GO_UP;
            stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_UP);
          }
        }
      } break;

      case STEP_DWELL: {
        // keep near Pmin during dwell between pressure steps
        driveToTarget(P_g, P_MIN_GAUGE_KPA, SPEED_RAMP_UP, SPEED_RAMP_DOWN);

        if (nowMs - dwellStartMs >= HOLD_BETWEEN_STEPS_MS) {
          float nextHigh = P_high_gauge + STEP_GAUGE_KPA;
          if (nextHigh > P_MAX_FINAL_GAUGE_KPA) nextHigh = P_MAX_FINAL_GAUGE_KPA;

          if (fabs(nextHigh - P_high_gauge) < 0.001f && P_high_gauge >= P_MAX_FINAL_GAUGE_KPA) {
            resetRamp();
            state = RAMP_POST;
            Serial.println(F("All fracture steps complete -> starting RAMP_POST (0->100 kPa_g)."));
          } else {
            P_high_gauge = nextHigh;
            cyclesThisStep = 0;
            highReached = false;
            state = GO_UP;
            stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_UP);

            Serial.print(F("Next step: Phigh -> "));
            Serial.print(P_high_gauge, 1);
            Serial.println(F(" kPa_g"));
          }
        }
      } break;

      case RAMP_POST: {
        bool done = runRampReachHold(P_g, nowMs, F("RAMP_POST"));
        if (done) {
          state = FINAL_RETURN_TO_ZERO;
          stepper.setSpeed(0.0f);
          Serial.println(F("RAMP_POST complete -> FINAL_RETURN_TO_ZERO."));
        }
      } break;

      case FINAL_RETURN_TO_ZERO: {
        bool done = runReturnToZero(P_g);
        if (done) {
          state = DONE;
          testRunning = false;
          stepper.setSpeed(0.0f);
          Serial.println(F("FINAL_RETURN_TO_ZERO complete. DONE."));
        }
      } break;

      default:
        stepper.setSpeed(0.0f);
        break;
    }
  } else {
    stepper.setSpeed(0.0f);
  }

  // Telemetry
  if (nowMs - lastPrintMs >= 150) {
    lastPrintMs = nowMs;

    Serial.print(F("P_g="));           Serial.print(P_g, 2);
    Serial.print(F(",Pmin_g="));       Serial.print(P_MIN_GAUGE_KPA, 1);
    Serial.print(F(",Phigh_g="));      Serial.print(P_high_gauge, 1);
    Serial.print(F(",rampTarget_g=")); Serial.print(rampTargetGauge, 1);
    Serial.print(F(",rampHold="));     Serial.print(rampInHold ? 1 : 0);
    Serial.print(F(",cyclesStep="));   Serial.print(cyclesThisStep);
    Serial.print(F(",posSteps="));     Serial.print(stepper.currentPosition());
    Serial.print(F(",DIR_SIGN="));     Serial.print(DIR_SIGN);

    Serial.print(F(",state="));
    switch (state) {
      case IDLE:                 Serial.print("IDLE"); break;
      case RETURN_TO_ZERO:       Serial.print("RETURN0"); break;
      case RAMP_PRE:             Serial.print("RAMP_PRE"); break;
      case GO_UP:                Serial.print("UP"); break;
      case HOLD_HIGH:            Serial.print("HOLD_HIGH"); break;
      case GO_DOWN:              Serial.print("DOWN"); break;
      case HOLD_LOW:             Serial.print("HOLD_LOW"); break;
      case STEP_DWELL:           Serial.print("DWELL"); break;
      case RAMP_POST:            Serial.print("RAMP_POST"); break;
      case FINAL_RETURN_TO_ZERO: Serial.print("FINAL_RETURN0"); break;
      case DONE:                 Serial.print("DONE"); break;
      default:                   Serial.print("UNK"); break;
    }

    Serial.print(F(",running="));      Serial.print(testRunning ? 1 : 0);
    Serial.println();
  }

  stepper.runSpeed();
}