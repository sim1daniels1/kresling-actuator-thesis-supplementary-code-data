// abp_fatigue_0_100g_500cycles_return_zero_end.ino
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

// ---- FATIGUE PROGRAM ----
const float P_MIN_GAUGE_KPA = 0.0f;
const float P_MAX_GAUGE_KPA = 100.0f;
long CYCLES_TARGET = 500;

// ---- Per-cycle dwells ----
const unsigned long HOLD_AT_HIGH_MS = 2000UL;
const unsigned long HOLD_AT_LOW_MS  = 2000UL;

// ---- Final return-to-zero after RAMP_POST ----
const unsigned long HOLD_AT_FINAL_ZERO_MS = 3000UL;

// ---- Control deadband for switching ----
float margin_kPa = 2.0f;

// ---- Hold control (gentle closed-loop) ----
const float HOLD_BAND_KPA = 2.0f;
float SPEED_HOLD_UP   = 250.0f;
float SPEED_HOLD_DOWN = -250.0f;

// ---- Speeds ----
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

// ===================== HARDWARE =====================

// ---- Pins ----
const int STEP_PIN  = 2;
const int DIR_PIN   = 5;
const int PRESS_PIN = A0;

// ---- Stepper ----
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ---- Pressure averaging ----
const int NAVG = 3;

// ===================== STATE MACHINE =====================

enum State {
  IDLE,
  RETURN_TO_ZERO,
  RAMP_PRE,
  GO_UP,
  HOLD_HIGH,
  GO_DOWN,
  HOLD_LOW,
  RAMP_POST,
  FINAL_RETURN_ZERO,
  FINAL_HOLD_ZERO,
  DONE
};

State state = IDLE;
bool testRunning = false;

// ---- Ramp vars ----
float rampTargetGauge = RAMP_BASE_GAUGE_KPA;
bool rampInHold = false;
unsigned long rampHoldStartMs = 0;

// ---- Fatigue vars ----
long cycleCount = 0;
bool highReachedThisCycle = false;
unsigned long cycleHoldStartMs = 0;

// ---- Final zero hold timer ----
unsigned long finalZeroHoldStartMs = 0;

// ============================================================================
// ABP 060PGAA5 transfer function: 10–90% Vs for 0..60 psi gauge (0..413.7 kPa_g)
// ============================================================================
const float P_GAUGE_FULLSCALE_KPA = 413.7f;

static inline float adcToGauge_kPa(int adc) {
  float r = adc / 1023.0f;
  float Pg = (r - 0.10f) / 0.80f * P_GAUGE_FULLSCALE_KPA;
  if (Pg < -10.0f) Pg = -10.0f;
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
// General target drive
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
// Gentle closed-loop hold
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
// Ramp reach -> hold -> step
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
      Serial.print(label);
      Serial.print(F(" reached "));
      Serial.print(rampTargetGauge, 1);
      Serial.println(F(" kPa_g -> HOLD"));
    }
  } else {
    if (!withinStay) {
      rampInHold = false;
      Serial.print(label);
      Serial.print(F(" drifted from "));
      Serial.print(rampTargetGauge, 1);
      Serial.println(F(" kPa_g -> re-acquire"));
    } else {
      if (nowMs - rampHoldStartMs >= HOLD_AT_TARGET_MS) {
        rampInHold = false;

        if (rampTargetGauge < RAMP_MAX_GAUGE_KPA) {
          rampTargetGauge += RAMP_STEP_GAUGE_KPA;
          if (rampTargetGauge > RAMP_MAX_GAUGE_KPA) rampTargetGauge = RAMP_MAX_GAUGE_KPA;

          Serial.print(label);
          Serial.print(F(" step -> "));
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
  cycleCount = 0;
  highReachedThisCycle = false;
  finalZeroHoldStartMs = 0;

  float sum_g = 0.0f;
  for (int i = 0; i < 30; i++) {
    sum_g += adcToGauge_kPa(analogRead(PRESS_PIN));
    delay(5);
  }
  float Pg_init = sum_g / 30.0f;

  testRunning = true;

  Serial.print(F("Fatigue test started. Pg_init="));
  Serial.print(Pg_init, 2);
  Serial.print(F(" kPa_g, cycles_target="));
  Serial.println(CYCLES_TARGET);

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
  if (fabs(SPEED_CYCLE_DOWN) > maxAbs) maxAbs = fabs(SPEED_CYCLE_DOWN);
  if (fabs(SPEED_HOLD_UP) > maxAbs)   maxAbs = fabs(SPEED_HOLD_UP);
  if (fabs(SPEED_HOLD_DOWN) > maxAbs) maxAbs = fabs(SPEED_HOLD_DOWN);

  stepper.setMaxSpeed(maxAbs + 200.0f);
  stepper.setAcceleration(1200.0f);
  stepper.setSpeed(0.0f);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  Serial.println(F("Rig ready (ABP fatigue 0-100 kPa_g, return-to-zero at end)."));
  Serial.println(F("Commands: 's' start, 'x' stop, 'i' invert dir, '5' set 500 cycles, '1' set 1000 cycles"));
}

// ============================================================================
// Loop
// ============================================================================
void loop() {
  static unsigned long lastPrintMs = 0;
  unsigned long nowMs = millis();

  // ---- Serial commands ----
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '5') {
      CYCLES_TARGET = 500;
      Serial.println(F("Cycles target set to 500."));
    } else if (c == '1') {
      CYCLES_TARGET = 1000;
      Serial.println(F("Cycles target set to 1000."));
    } else if (c == 'i' || c == 'I') {
      DIR_SIGN = -DIR_SIGN;
      Serial.print(F("Direction inverted. DIR_SIGN="));
      Serial.println(DIR_SIGN);
    } else if ((c == 's' || c == 'S') && !testRunning) {
      startTest();
    } else if (c == 'x' || c == 'X') {
      stopTest(F("Test stopped."));
    }
  }

  // ---- Pressure read ----
  float P_g = readGauge_kPa();
  if (isnan(P_g)) {
    stepper.runSpeed();
    return;
  }

  // ---- Safety ----
  if (testRunning && P_g >= P_HARD_MAX_GAUGE_KPA) {
    stopTest(F("SAFETY STOP: gauge pressure exceeded hard limit."));
  }

  // ---- State machine ----
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
          state = GO_DOWN;
          highReachedThisCycle = true;
          stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_DOWN);
          Serial.println(F("RAMP_PRE complete -> starting fatigue cycling 0 <-> 100 kPa_g."));
        }
      } break;

      case GO_UP: {
        if (P_g < P_MAX_GAUGE_KPA - margin_kPa) stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_UP);
        else stepper.setSpeed(0.0f);

        if (P_g >= P_MAX_GAUGE_KPA - margin_kPa) {
          highReachedThisCycle = true;
          state = HOLD_HIGH;
          cycleHoldStartMs = nowMs;
          Serial.println(F("Reached Pmax -> HOLD_HIGH"));
        }
      } break;

      case HOLD_HIGH: {
        holdToTarget(P_g, P_MAX_GAUGE_KPA);

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
        holdToTarget(P_g, P_MIN_GAUGE_KPA);

        if (nowMs - cycleHoldStartMs >= HOLD_AT_LOW_MS) {
          if (highReachedThisCycle) {
            cycleCount++;
            highReachedThisCycle = false;

            Serial.print(F("Cycle "));
            Serial.print(cycleCount);
            Serial.print(F("/"));
            Serial.print(CYCLES_TARGET);
            Serial.println(F(" complete."));
          }

          if (cycleCount >= CYCLES_TARGET) {
            resetRamp();
            state = RAMP_POST;
            Serial.println(F("FATIGUE complete -> starting RAMP_POST (0->100 kPa_g)."));
          } else {
            state = GO_UP;
            stepper.setSpeed(DIR_SIGN * SPEED_CYCLE_UP);
          }
        }
      } break;

      case RAMP_POST: {
        bool done = runRampReachHold(P_g, nowMs, F("RAMP_POST"));
        if (done) {
          state = FINAL_RETURN_ZERO;
          Serial.println(F("RAMP_POST complete -> returning to 0 kPa_g."));
        }
      } break;

      case FINAL_RETURN_ZERO: {
        bool done = runReturnToZero(P_g);
        if (done) {
          state = FINAL_HOLD_ZERO;
          finalZeroHoldStartMs = nowMs;
          Serial.println(F("Reached final 0 kPa_g -> FINAL_HOLD_ZERO"));
        }
      } break;

      case FINAL_HOLD_ZERO: {
        holdToTarget(P_g, 0.0f);

        if (nowMs - finalZeroHoldStartMs >= HOLD_AT_FINAL_ZERO_MS) {
          state = DONE;
          testRunning = false;
          stepper.setSpeed(0.0f);
          Serial.println(F("FINAL_HOLD_ZERO complete. DONE."));
        }
      } break;

      case DONE:
      case IDLE:
      default:
        stepper.setSpeed(0.0f);
        break;
    }
  } else {
    stepper.setSpeed(0.0f);
  }

  // ---- Telemetry ----
  if (nowMs - lastPrintMs >= 150) {
    lastPrintMs = nowMs;

    Serial.print(F("P_g="));            Serial.print(P_g, 2);
    Serial.print(F(",Pmin_g="));        Serial.print(P_MIN_GAUGE_KPA, 1);
    Serial.print(F(",Pmax_g="));        Serial.print(P_MAX_GAUGE_KPA, 1);
    Serial.print(F(",rampTarget_g="));  Serial.print(rampTargetGauge, 1);
    Serial.print(F(",rampHold="));      Serial.print(rampInHold ? 1 : 0);
    Serial.print(F(",cycles="));        Serial.print(cycleCount);
    Serial.print(F(",cyclesTgt="));     Serial.print(CYCLES_TARGET);
    Serial.print(F(",posSteps="));      Serial.print(stepper.currentPosition());
    Serial.print(F(",DIR_SIGN="));      Serial.print(DIR_SIGN);

    Serial.print(F(",state="));
    switch (state) {
      case IDLE:              Serial.print("IDLE"); break;
      case RETURN_TO_ZERO:    Serial.print("RETURN0"); break;
      case RAMP_PRE:          Serial.print("RAMP_PRE"); break;
      case GO_UP:             Serial.print("UP"); break;
      case HOLD_HIGH:         Serial.print("HOLD_HIGH"); break;
      case GO_DOWN:           Serial.print("DOWN"); break;
      case HOLD_LOW:          Serial.print("HOLD_LOW"); break;
      case RAMP_POST:         Serial.print("RAMP_POST"); break;
      case FINAL_RETURN_ZERO: Serial.print("FINAL_RETURN0"); break;
      case FINAL_HOLD_ZERO:   Serial.print("FINAL_HOLD0"); break;
      case DONE:              Serial.print("DONE"); break;
      default:                Serial.print("UNK"); break;
    }

    Serial.print(F(",running="));       Serial.print(testRunning ? 1 : 0);
    Serial.println();
  }

  stepper.runSpeed();
}