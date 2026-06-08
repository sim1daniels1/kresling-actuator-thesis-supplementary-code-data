// manual_pressure_controller_mpx4250.ino
#include <AccelStepper.h>

// ---------- Pins ----------
const int STEP_PIN = 2;
const int DIR_PIN  = 5;
const int SENSOR_PIN = A1;   // MPX4250AP used on A1 earlier

// ---------- MPX4250AP absolute sensor ----------
const float ADC_MAX = 1023.0;

// Datasheet transfer function:
// Vout/Vs = 0.004 * P_abs - 0.04
// Therefore:
// P_abs = (Vout/Vs + 0.04) / 0.004
//       = 250 * (Vout/Vs + 0.04)
float readPressureAbsKpa() {
  int adc = analogRead(SENSOR_PIN);
  float ratio = adc / ADC_MAX;
  float P_abs = 250.0 * (ratio + 0.04);
  return P_abs;
}

// ---------- Stepper ----------
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

int DIR_SIGN = -1;

const float MOVE_SPEED = 500.0;
const float HOLD_SPEED = 100.0;

// ---------- Control ----------
float P_atm = 101.0;
float target_gauge_kPa = 0.0;

const float STEP_KPA = 50.0;
const float MIN_GAUGE_KPA = -100.0;
const float MAX_GAUGE_KPA = 200.0;

const float ENTER_BAND_KPA = 5.0;
const float HOLD_BAND_KPA  = 10.0;

bool running = false;
String state = "IDLE";

unsigned long lastLog = 0;
const unsigned long LOG_INTERVAL_MS = 100;

// ---------- Sensor averaging ----------
float readAvgAbsKpa(int n = 30) {
  float sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += readPressureAbsKpa();
    delay(5);
  }
  return sum / n;
}

// ---------- Helpers ----------
void startMoveToTarget() {
  running = true;
  state = "MOVING";
}

void stopMotor() {
  stepper.setSpeed(0);
  running = false;
  state = "HOLD";
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd == "UP") {
    target_gauge_kPa += STEP_KPA;
    if (target_gauge_kPa > MAX_GAUGE_KPA) target_gauge_kPa = MAX_GAUGE_KPA;
    startMoveToTarget();
  }

  else if (cmd == "DOWN") {
    target_gauge_kPa -= STEP_KPA;
    if (target_gauge_kPa < MIN_GAUGE_KPA) target_gauge_kPa = MIN_GAUGE_KPA;
    startMoveToTarget();
  }

  else if (cmd == "ZERO") {
    target_gauge_kPa = 0.0;
    startMoveToTarget();
  }

  else if (cmd == "STOP") {
    stepper.setSpeed(0);
    running = false;
    state = "IDLE";
  }

  else if (cmd == "I") {
    DIR_SIGN *= -1;
  }

  else if (cmd == "REZERO") {
    P_atm = readAvgAbsKpa(50);
    target_gauge_kPa = 0.0;
    state = "IDLE";
    running = false;
  }

  else if (cmd.startsWith("SET ")) {
    float val = cmd.substring(4).toFloat();
    target_gauge_kPa = constrain(val, MIN_GAUGE_KPA, MAX_GAUGE_KPA);
    startMoveToTarget();
  }
}

void setup() {
  Serial.begin(115200);

  stepper.setMaxSpeed(2000);
  stepper.setAcceleration(3000);

  delay(1000);

  // Estimate atmospheric pressure at startup
  // Make sure the system is vented to atmosphere here.
  P_atm = readAvgAbsKpa(50);
  target_gauge_kPa = 0.0;

  Serial.println("timestamp_ms,P_abs,P_g,P_atm,target_gauge_kPa,state,posSteps,DIR_SIGN");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  float P_abs = readPressureAbsKpa();
  float P_g = P_abs - P_atm;
  float error = target_gauge_kPa - P_g;

  // ---------- Closed-loop movement ----------
  if (running) {
    if (abs(error) <= ENTER_BAND_KPA) {
      stopMotor();
    } else {
      if (error > 0) {
        stepper.setSpeed(DIR_SIGN * MOVE_SPEED);
      } else {
        stepper.setSpeed(-DIR_SIGN * MOVE_SPEED);
      }

      stepper.runSpeed();
    }
  }

  else if (state == "HOLD") {
    if (abs(error) > HOLD_BAND_KPA) {
      if (error > 0) {
        stepper.setSpeed(DIR_SIGN * HOLD_SPEED);
      } else {
        stepper.setSpeed(-DIR_SIGN * HOLD_SPEED);
      }

      stepper.runSpeed();
    } else {
      stepper.setSpeed(0);
    }
  }

  // ---------- Safety ----------
  if (P_g > MAX_GAUGE_KPA + 20.0) {
    stepper.setSpeed(0);
    running = false;
    state = "SAFETY_STOP";
  }

  // ---------- Logging ----------
  unsigned long now = millis();
  if (now - lastLog >= LOG_INTERVAL_MS) {
    lastLog = now;

    Serial.print(now);
    Serial.print(",");
    Serial.print(P_abs, 2);
    Serial.print(",");
    Serial.print(P_g, 2);
    Serial.print(",");
    Serial.print(P_atm, 2);
    Serial.print(",");
    Serial.print(target_gauge_kPa, 2);
    Serial.print(",");
    Serial.print(state);
    Serial.print(",");
    Serial.print(stepper.currentPosition());
    Serial.print(",");
    Serial.println(DIR_SIGN);
  }
}