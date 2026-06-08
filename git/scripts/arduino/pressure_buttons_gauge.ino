// manual_pressure_controller_abp.ino
#include <AccelStepper.h>

// ---------- Pins ----------
const int STEP_PIN = 2;
const int DIR_PIN  = 5;
const int SENSOR_PIN = A0;

// ---------- Sensor: Honeywell ABP 060PGAA5 ----------
const float FULL_SCALE_KPA = 413.7;   // 60 psi
const float ADC_MAX = 1023.0;

// Transfer Function A: 10–90% Vs
float readPressureGaugeKpa() {
  int adc = analogRead(SENSOR_PIN);
  float frac = adc / ADC_MAX;
  float p = ((frac - 0.10) / 0.80) * FULL_SCALE_KPA;
  return p;
}

// ---------- Stepper ----------
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// Change this if pressure moves the wrong way
int DIR_SIGN = -1;

// Speeds
const float MOVE_SPEED = 500.0;      // microsteps/s
const float HOLD_SPEED = 50.0;       // gentle correction speed

// ---------- Control ----------
float target_kPa = 0.0;
const float STEP_KPA = 50.0;
const float MIN_KPA = 0.0;
const float MAX_KPA = 320.0;

const float ENTER_BAND_KPA = 1.0;     // stop when close to target
const float HOLD_BAND_KPA  = 2.0;     // correct if drifting outside this

bool running = false;
String state = "IDLE";

unsigned long lastLog = 0;
const unsigned long LOG_INTERVAL_MS = 100;

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
    target_kPa += STEP_KPA;
    if (target_kPa > MAX_KPA) target_kPa = MAX_KPA;
    startMoveToTarget();
  }

  else if (cmd == "DOWN") {
    target_kPa -= STEP_KPA;
    if (target_kPa < MIN_KPA) target_kPa = MIN_KPA;
    startMoveToTarget();
  }

  else if (cmd == "ZERO") {
    target_kPa = 0.0;
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

  else if (cmd.startsWith("SET ")) {
    float val = cmd.substring(4).toFloat();
    target_kPa = constrain(val, MIN_KPA, MAX_KPA);
    startMoveToTarget();
  }
}

void setup() {
  Serial.begin(115200);

  stepper.setMaxSpeed(2000);
  stepper.setAcceleration(3000);

  target_kPa = readPressureGaugeKpa();
  if (target_kPa < 0) target_kPa = 0;

  Serial.println("timestamp_ms,P_g,target_kPa,state,posSteps,DIR_SIGN");
}

void loop() {
  // ---------- Read serial ----------
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  float P_g = readPressureGaugeKpa();
  float error = target_kPa - P_g;

  // ---------- Control logic ----------
  if (running) {
    if (abs(error) <= ENTER_BAND_KPA) {
      stopMotor();
    } else {
      float speed = MOVE_SPEED;

      if (error > 0) {
        stepper.setSpeed(DIR_SIGN * speed);
      } else {
        stepper.setSpeed(-DIR_SIGN * speed);
      }

      stepper.runSpeed();
    }
  }

  else if (state == "HOLD") {
    if (abs(error) > HOLD_BAND_KPA) {
      float speed = HOLD_SPEED;

      if (error > 0) {
        stepper.setSpeed(DIR_SIGN * speed);
      } else {
        stepper.setSpeed(-DIR_SIGN * speed);
      }

      stepper.runSpeed();
    } else {
      stepper.setSpeed(0);
    }
  }

  // ---------- Safety ----------
  if (P_g > MAX_KPA + 20.0) {
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
    Serial.print(P_g, 2);
    Serial.print(",");
    Serial.print(target_kPa, 2);
    Serial.print(",");
    Serial.print(state);
    Serial.print(",");
    Serial.print(stepper.currentPosition());
    Serial.print(",");
    Serial.println(DIR_SIGN);
  }
}