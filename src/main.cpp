#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <ESP_I2S.h>
#include <DFRobotDFPlayerMini.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <esp_heap_caps.h>
#include <esp_sleep.h>
#include <math.h>
#include <string.h>
#include <NAS_Classification_fi_inferencing.h>


const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* TB_HOST = "YOUR_THINGSBOARD_HOST";
const char* TB_TOKEN = "YOUR_DEVICE_ACCESS_TOKEN";
constexpr uint16_t TB_PORT = 1883;

constexpr bool ENABLE_DEEP_SLEEP = true;


constexpr int MIC_BCLK = 8;
constexpr int MIC_WS   = 9;
constexpr int MIC_SD   = 10;
constexpr int OLED_SDA = 16;
constexpr int OLED_SCL = 15;


constexpr int LED1 = 6;
constexpr int LED2 = 5;
constexpr int LED3 = 4;
constexpr int RADAR_OUT = 11;


constexpr int RADAR_RX = 18;
constexpr int RADAR_TX = 17;

constexpr int DF_RX = 2;
constexpr int DF_TX = 1;
constexpr int DF_BUSY = 12;


constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t BLOCK_SIZE = 256;
constexpr size_t DB_WINDOW = SAMPLE_RATE / 2;
constexpr size_t AUDIO_WINDOW = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
constexpr float MIC_GAIN = 1.0f;
constexpr float DB_OFFSET = -17.0f;

const double REF_RMS_PCM16 = (420426.0 / sqrt(2.0)) / 256.0;


constexpr float DEFAULT_CRITICAL_DB = 85.0f;
constexpr float MIN_CRITICAL_DB = 80.0f;
constexpr float MAX_CRITICAL_DB = 110.0f;

constexpr char TB_CRITICAL_KEY[] = "critical_db";


constexpr float WEB_MIN_NOISE_DB = 0.0f;
constexpr float WEB_MAX_NOISE_DB = 140.0f;
constexpr char WEB_THRESHOLD_KEY[] = "noise_limit";
constexpr char WEB_CONFIG_VERSION_KEY[] = "config_version";
constexpr uint32_t WEB_ACK_RETRY_MS = 10000;
constexpr uint32_t TELEMETRY_MS = 30000;
constexpr uint32_t SLEEP_AFTER_MS = 30000;
constexpr uint32_t VOICE_COOLDOWN_MS = 30000;
constexpr uint32_t VOICE_SETTLE_MS = 500;
constexpr uint32_t VOICE_TIMEOUT_MS = 15000;
constexpr uint8_t DF_VOLUME = 22;

constexpr uint16_t TRACK_CRITICAL = 2;


I2SClass mic;
HardwareSerial dfPort(1);
DFRobotDFPlayerMini dfplayer;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
WiFiClient netClient;
PubSubClient mqtt(netClient);
Preferences preferences;
float criticalDb = DEFAULT_CRITICAL_DB;

bool webConfigActive = false;
uint64_t webAppliedVersion = 0;
bool webAckPending = false;
uint32_t webLastAckAttempt = 0;
bool aiHasResult = false;
bool audioHasReading = false;

int32_t rawBlock[BLOCK_SIZE];
int16_t* pcmBuffer = nullptr;
size_t sampleIndex = 0;
size_t dbSamples = 0;
double dbSum = 0.0, dbSquares = 0.0;
float currentDb = 0.0f;
float currentAcRms = 0.0f;
float sumDbPeriod = 0.0f, maxDbPeriod = -1000.0f;
uint32_t nDbPeriod = 0;
char detectedClass[24] = "Unknown";
float confidence = 0.0f;

bool oledReady = false;
bool micReady = false;
bool dfReady = false;
bool voicePlaying = false;
bool voiceSettling = false;
bool voiceBusySeenLow = false;
uint16_t activeVoiceTrack = 0;
bool radarLast = false;
bool pendingCritical = false;
bool criticalSentOnce = false;
uint32_t lastCriticalEventAt = 0;
uint32_t noPersonFrom = 0;
uint32_t voiceStartedAt = 0;
uint32_t voiceFinishedAt = 0;
uint32_t lastTelemetryAt = 0;
uint32_t lastWiFiTryAt = 0;
uint32_t lastMqttTryAt = 0;


void queueCriticalAtNoiseLimit(const char* source) {
  if (!audioHasReading || !isfinite(currentDb) || currentDb < criticalDb) return;
  if (criticalSentOnce && millis() - lastCriticalEventAt < VOICE_COOLDOWN_MS) return;
  if (pendingCritical) return;
  pendingCritical = true;
  criticalSentOnce = true;
  lastCriticalEventAt = millis();
  Serial.printf("CRITICAL (%s): measured %.1f >= noise_limit %.1f dB; DFPlayer /mp3/0002.mp3 queued\n",
                source, currentDb, criticalDb);
}

bool isPresent() { return digitalRead(RADAR_OUT) == HIGH; }
bool wifiSettingsValid() {
  return strcmp(WIFI_SSID, "YOUR_WIFI_SSID") != 0 &&
         strcmp(TB_HOST, "YOUR_THINGSBOARD_HOST") != 0 &&
         strcmp(TB_TOKEN, "YOUR_DEVICE_ACCESS_TOKEN") != 0;
}

void ledsOff() {
  digitalWrite(LED1, HIGH);
  digitalWrite(LED2, HIGH);
  digitalWrite(LED3, HIGH);
}
void updateLeds(float db) {
  digitalWrite(LED1, db > 75.0f ? LOW : HIGH);
  digitalWrite(LED2, db >= 80.0f ? LOW : HIGH);
  digitalWrite(LED3, db >= criticalDb ? LOW : HIGH);
}

void displayScreen(const char* status) {
  if (!oledReady) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.print("NAS V2  ");
  oled.println(status);
  oled.drawLine(0, 11, 127, 11, SSD1306_WHITE);
  oled.setCursor(0, 15);
  oled.setTextSize(2);
  oled.print(currentDb, 1);
  oled.setTextSize(1);
  oled.println(" dB*");
  oled.setCursor(0, 39);
  oled.print("Class: "); oled.println(detectedClass);
  oled.setCursor(0, 49);
  oled.print("Score: "); oled.print(confidence * 100.0f, 1); oled.println("%");
  oled.setCursor(0, 58);
  oled.print("P:"); oled.print(isPresent() ? "YES" : "NO");
  oled.print(" Alert:"); oled.print("ON");
  oled.display();
}

void resetCapture() {
  sampleIndex = 0;
  dbSamples = 0;
  dbSum = 0;
  dbSquares = 0;
}

bool startMic() {
  if (micReady) return true;
  mic.setPins(MIC_BCLK, MIC_WS, -1, MIC_SD);
  if (!mic.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("ERROR: INMP441 I2S start failed");
    return false;
  }
  micReady = true;
  return true;
}
void stopMic() {
  if (micReady) {
    mic.end();
    micReady = false;
  }
}
void warmupMic() {
  if (!micReady) return;
  size_t remaining = SAMPLE_RATE / 2;
  while (remaining > 0) {
    size_t target = remaining < BLOCK_SIZE ? remaining : BLOCK_SIZE;
    size_t bytesRead = mic.readBytes((char*)rawBlock, target * sizeof(int32_t));
    if (bytesRead == 0 || bytesRead % sizeof(int32_t) != 0) {
      Serial.println("WARNING: mic warmup read error");
      return;
    }
    remaining -= bytesRead / sizeof(int32_t);
  }
}

bool publishTelemetry(const char* json) {
  if (!mqtt.connected()) {
    Serial.println("ThingsBoard: offline, not sent");
    return false;
  }
  bool ok = mqtt.publish("v1/devices/me/telemetry", json);
  Serial.printf("ThingsBoard: %s | %s\n", ok ? "sent" : "failed", json);
  return ok;
}
void sendPresence(bool p) {
  char json[72];
  snprintf(json, sizeof(json), "{\"presence\":%s}", p ? "true" : "false");
  publishTelemetry(json);
}
void sendEvent(const char* event) {
  char json[220];
  snprintf(json, sizeof(json),
    "{\"event\":\"%s\",\"sound_db\":%.1f,\"sound_class\":\"%s\",\"presence\":%s}",
    event, currentDb, detectedClass, isPresent() ? "true" : "false");
  publishTelemetry(json);
}
void sendEvery30s() {
  float mean = nDbPeriod ? sumDbPeriod / nDbPeriod : currentDb;
  float maximum = nDbPeriod ? maxDbPeriod : currentDb;
  char json[290];
  snprintf(json, sizeof(json),
    "{\"sound_db\":%.1f,\"sound_db_avg\":%.1f,\"sound_db_max\":%.1f,"
    "\"sound_class\":\"%s\",\"confidence\":%.3f,\"presence\":%s,"
    "\"speech_with_presence\":%s,\"alert_muted\":%s,\"critical_db\":%.1f}",
    currentDb, mean, maximum, detectedClass, confidence,
    isPresent() ? "true" : "false",
    (isPresent() && strcmp(detectedClass, "Speech") == 0) ? "true" : "false",
    "false", criticalDb);
  publishTelemetry(json);


  const char* websiteSoundType = "Unknown";
  if (aiHasResult) {
    if (strcmp(detectedClass, "Speech") == 0) websiteSoundType = "Speech";
    else if (strcmp(detectedClass, "Music") == 0) websiteSoundType = "Music / Media";
    else if (strcmp(detectedClass, "Noise") == 0) websiteSoundType = "Background Noise";
  }


  bool validAudio = audioHasReading && isfinite(currentDb) && micReady;
  bool currentAlert = validAudio && currentDb >= criticalDb;
  char webJson[360];
  if (aiHasResult) {
    snprintf(webJson, sizeof(webJson),
      "{\"sound_db\":%.1f,\"sound_type\":\"%s\",\"confidence\":%.3f,"
      "\"noise_limit\":%.1f,\"alert\":%s,\"presence\":%s,"
      "\"audio_valid\":%s,\"device_status\":\"awake\"}",
      currentDb, websiteSoundType, confidence, criticalDb,
      currentAlert ? "true" : "false", isPresent() ? "true" : "false",
      validAudio ? "true" : "false");
  } else {
    snprintf(webJson, sizeof(webJson),
      "{\"sound_db\":%.1f,\"sound_type\":\"Unknown\","
      "\"noise_limit\":%.1f,\"alert\":%s,\"presence\":%s,"
      "\"audio_valid\":%s,\"device_status\":\"awake\"}",
      currentDb, criticalDb, currentAlert ? "true" : "false",
      isPresent() ? "true" : "false", validAudio ? "true" : "false");
  }
  publishTelemetry(webJson);
  sumDbPeriod = 0;
  maxDbPeriod = -1000;
  nDbPeriod = 0;
}


void sendWebsiteAck() {
  if (!webAckPending || !mqtt.connected() || webAppliedVersion == 0) return;
  char ack[96];
  snprintf(ack, sizeof(ack),
           "{\"applied_config_version\":%llu}",
           (unsigned long long)webAppliedVersion);
  webLastAckAttempt = millis();
  if (mqtt.publish("v1/devices/me/attributes", ack)) {
    webAckPending = false;
    Serial.printf("Website: configuration ACK published version=%llu\n",
                  (unsigned long long)webAppliedVersion);
  } else {
    Serial.println("Website: ACK failed; retry later");
  }
}

bool applyWebsiteConfig(JsonVariantConst container) {
  JsonVariantConst thresholdVariant = container[WEB_THRESHOLD_KEY];
  JsonVariantConst versionVariant = container[WEB_CONFIG_VERSION_KEY];
  if (thresholdVariant.isNull() || versionVariant.isNull()) return false;

  if (!(thresholdVariant.is<int>() || thresholdVariant.is<long>() ||
        thresholdVariant.is<float>() || thresholdVariant.is<double>())) {
    Serial.println("Website: noise_limit must be numeric");
    return true;
  }
  float candidate = thresholdVariant.as<float>();
  if (!isfinite(candidate) || candidate < WEB_MIN_NOISE_DB || candidate > WEB_MAX_NOISE_DB) {
    Serial.printf("Website: rejected noise_limit=%.2f (range 0..140)\n", candidate);
    return true;
  }
  if (!(versionVariant.is<uint64_t>() || versionVariant.is<int64_t>() ||
        versionVariant.is<unsigned long>() || versionVariant.is<long>())) {
    Serial.println("Website: config_version must be integer milliseconds");
    return true;
  }
  uint64_t version = versionVariant.as<uint64_t>();
  if (version == 0) {
    Serial.println("Website: rejected empty config_version");
    return true;
  }

  if (webConfigActive && version < webAppliedVersion) {
    Serial.println("Website: ignored old config version");
    return true;
  }


  size_t storedDb = preferences.putFloat("webNoiseDb", candidate);
  size_t storedVersion = preferences.putULong64("webCfgVer", version);
  if (!storedDb || !storedVersion) {
    Serial.println("Website: NVS save failed; NO ACK");
    return true;
  }
  criticalDb = candidate;
  webAppliedVersion = version;
  webConfigActive = true;
  webAckPending = true;
  updateLeds(currentDb);


  queueCriticalAtNoiseLimit("website update");
  displayScreen("WEB CONFIG");
  Serial.printf("Website: applied noise_limit=%.1f dB, version=%llu\n",
                criticalDb, (unsigned long long)webAppliedVersion);
  sendWebsiteAck();
  return true;
}


void applyCriticalDb(float candidate) {
  if (!isfinite(candidate) || candidate < MIN_CRITICAL_DB || candidate > MAX_CRITICAL_DB) {
    Serial.printf("ThingsBoard: ignored invalid critical_db %.2f (allowed %.0f..%.0f)\n",
                  candidate, MIN_CRITICAL_DB, MAX_CRITICAL_DB);
    return;
  }
  if (fabsf(candidate - criticalDb) < 0.01f) return;
  criticalDb = candidate;
  preferences.putFloat("criticalDb", criticalDb);
  updateLeds(currentDb);
  Serial.printf("ThingsBoard: CRITICAL THRESHOLD UPDATED to %.1f dB\n", criticalDb);
  displayScreen("THRESHOLD SET");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {

  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) { Serial.printf("ThingsBoard attribute JSON error: %s\n", err.c_str()); return; }

  if (applyWebsiteConfig(doc.as<JsonVariantConst>())) return;
  if (applyWebsiteConfig(doc["shared"].as<JsonVariantConst>())) return;

  if (webConfigActive) return;
  JsonVariant v = doc[TB_CRITICAL_KEY];
  if (v.isNull()) v = doc["shared"][TB_CRITICAL_KEY];
  if (v.isNull()) return;
  if (!(v.is<float>() || v.is<double>() || v.is<int>() || v.is<long>())) {
    Serial.println("ThingsBoard: critical_db must be a NUMBER, not text");
    return;
  }
  applyCriticalDb(v.as<float>());
}

void serviceNetwork() {
  if (!wifiSettingsValid()) return;
  uint32_t now = millis();
  if (WiFi.status() != WL_CONNECTED) {
    if (now - lastWiFiTryAt >= 15000) {
      lastWiFiTryAt = now;
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      Serial.println("WiFi reconnect requested");
    }
    return;
  }
  if (!mqtt.connected() && now - lastMqttTryAt >= 15000) {
    lastMqttTryAt = now;
    String id = "nas-v2-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    bool ok = mqtt.connect(id.c_str(), TB_TOKEN, nullptr);
    Serial.println(ok ? "ThingsBoard CONNECTED" : "ThingsBoard connect failed");
    if (ok) {
      bool subUpdates = mqtt.subscribe("v1/devices/me/attributes");
      bool subResponses = mqtt.subscribe("v1/devices/me/attributes/response/+");
      bool requested = mqtt.publish("v1/devices/me/attributes/request/1",
                                    "{\"sharedKeys\":\"critical_db\"}");
      Serial.printf("ThingsBoard: shared attributes subscribe=%d response=%d request=%d\n",
                    subUpdates, subResponses, requested);

      bool webRequested = mqtt.publish("v1/devices/me/attributes/request/2",
                                     "{\"sharedKeys\":\"noise_limit,config_version\"}");
      Serial.printf("Website: requested noise_limit/config_version = %d\n", webRequested);
      if (webConfigActive) {
        webAckPending = true;
        sendWebsiteAck();
      }
    }
  }
  if (mqtt.connected()) {
    mqtt.loop();
    if (webAckPending && millis() - webLastAckAttempt >= WEB_ACK_RETRY_MS)
      sendWebsiteAck();
  }
}

void endVoice() {
  if (!voicePlaying) return;
  voicePlaying = false;
  voiceSettling = true;
  voiceFinishedAt = millis();
  Serial.println("DFPlayer finished; audio settling...");
}

bool startVoice(uint16_t track) {
  if (!dfReady || voicePlaying || voiceSettling) {
    Serial.println("DFPlayer unavailable or busy; no voice playback");
    return false;
  }
  stopMic();
  resetCapture();
  voicePlaying = true;
  voiceBusySeenLow = false;
  activeVoiceTrack = track;
  voiceStartedAt = millis();
  displayScreen("VOICE");
  dfplayer.playMp3Folder(track);
  Serial.printf("DFPlayer: /mp3/%04u.mp3\n", (unsigned)track);
  return true;
}

void serviceVoice() {
  if (voicePlaying) {
    if (DF_BUSY >= 0) {

      if (digitalRead(DF_BUSY) == LOW) voiceBusySeenLow = true;
      if (voiceBusySeenLow && digitalRead(DF_BUSY) == HIGH &&
          millis() - voiceStartedAt > 300) endVoice();
    } else {
      while (dfplayer.available()) {
        uint8_t eventType = dfplayer.readType();
        (void)dfplayer.read();
        if (eventType == DFPlayerPlayFinished && millis() - voiceStartedAt > 500)
          endVoice();
      }
    }
    if (voicePlaying && millis() - voiceStartedAt >= VOICE_TIMEOUT_MS) {
      Serial.println("WARNING: DFPlayer finish not detected, using 15s timeout");
      endVoice();
    }
  }
  if (voiceSettling && millis() - voiceFinishedAt >= VOICE_SETTLE_MS) {
    voiceSettling = false;
    resetCapture();
    if (startMic()) {
      warmupMic();
      Serial.println("Microphone RESUMED; fresh AI window");
      displayScreen("LISTENING");
    } else {
      displayScreen("MIC ERROR");
    }
  }
}

void goToSleep() {
  if (!ENABLE_DEEP_SLEEP || isPresent() || voicePlaying || voiceSettling) return;
  Serial.println("No person 30s -> Deep Sleep (GPIO11 HIGH wake)");

  esp_err_t err = esp_sleep_enable_ext1_wakeup(1ULL << RADAR_OUT, ESP_EXT1_WAKEUP_ANY_HIGH);
  if (err != ESP_OK) {
    Serial.printf("Sleep wake configuration failed: %d\n", (int)err);
    noPersonFrom = millis();
    return;
  }
  delay(100);
  if (isPresent()) {
    noPersonFrom = 0;
    return;
  }
  sendPresence(false);

  publishTelemetry("{\"device_status\":\"sleeping\",\"presence\":false}");
  mqtt.loop();
  mqtt.disconnect();
  WiFi.disconnect(true);
  stopMic();
  ledsOff();
  if (oledReady) { oled.clearDisplay(); oled.display(); }

  if (isPresent()) {
    Serial.println("Person returned during shutdown - cancelling sleep");
    noPersonFrom = 0;
    if (startMic()) warmupMic();
    if (oledReady) displayScreen("LISTENING");
    return;
  }
  Serial.println("Entering deep sleep now (EXT1, GPIO11 HIGH wake)");
  Serial.flush();
  delay(200);
  esp_deep_sleep_start();
}
void monitorRadar() {
  uint32_t now = millis();
  bool present = isPresent();
  if (present != radarLast) {
    radarLast = present;
    Serial.printf("Radar OUT GPIO11 = %d (%s)\n", present, present ? "PRESENT" : "ABSENT");
    sendPresence(present);
    if (present) {
      noPersonFrom = 0;
      resetCapture();
    } else {
      noPersonFrom = now;
    }
    displayScreen("RADAR");
  }
  if (!present && noPersonFrom != 0 && now - noPersonFrom >= SLEEP_AFTER_MS)
    goToSleep();
}


void updateDbWindow() {
  if (!dbSamples) return;
  double mean = dbSum / dbSamples;
  double acPower = dbSquares / dbSamples - mean * mean;
  double acRms = sqrt(acPower > 0 ? acPower : 0.0);
  currentAcRms = (float)acRms;
  double rawDb = 94.0 + 20.0 * log10(fmax(acRms, 0.000001) / REF_RMS_PCM16);
  currentDb = (float)(rawDb + DB_OFFSET);
  audioHasReading = true;

  Serial.printf("dB*: %.1f | Raw dB: %.1f | AC RMS: %.1f | Radar: %d\n",
                currentDb, rawDb, currentAcRms, isPresent());
  sumDbPeriod += currentDb;
  nDbPeriod++;
  if (currentDb > maxDbPeriod) maxDbPeriod = currentDb;
  updateLeds(currentDb);
  displayScreen("LISTENING");


  queueCriticalAtNoiseLimit("microphone");
  dbSamples = 0;
  dbSum = 0;
  dbSquares = 0;
}
void processAlerts() {
  if (pendingCritical) {
    pendingCritical = false;
    displayScreen("CRITICAL");
    Serial.println("EVENT CRITICAL");

    startVoice(TRACK_CRITICAL);
    sendEvent("critical");
  }
}

static int getSignal(size_t offset, size_t length, float* dest) {
  if (pcmBuffer == nullptr || offset > AUDIO_WINDOW || length > AUDIO_WINDOW - offset)
    return -1;
  for (size_t i = 0; i < length; ++i) dest[i] = (float)pcmBuffer[offset + i];
  return 0;
}
void inferAudio() {
  signal_t signal;
  signal.total_length = AUDIO_WINDOW;
  signal.get_data = getSignal;
  ei_impulse_result_t result = {0};
  displayScreen("AI WORKING");
  Serial.println("AI analyzing 5-second audio...");
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.printf("ERROR: AI run_classifier=%d\n", (int)err);
    displayScreen("AI ERROR");
    return;
  }
  float best = -1;
  const char* bestLabel = "Unknown";
  Serial.println("--- AI RESULT ---");
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; ++i) {
    const float score = result.classification[i].value;
    Serial.printf("%s = %.1f%%\n", result.classification[i].label, score * 100.0f);
    if (score > best) { best = score; bestLabel = result.classification[i].label; }
  }
  strncpy(detectedClass, bestLabel, sizeof(detectedClass) - 1);
  detectedClass[sizeof(detectedClass) - 1] = '\0';
  confidence = best;
  aiHasResult = true;
  Serial.printf("Detected=%s | Confidence=%.1f%% | DSP=%d ms | NN=%d ms\n",
    detectedClass, confidence * 100.0f,
    result.timing.dsp, result.timing.classification);
  displayScreen("AI READY");
}

void collectMicChunk() {
  if (!micReady || pcmBuffer == nullptr) return;
  size_t wanted = AUDIO_WINDOW - sampleIndex;
  if (wanted > BLOCK_SIZE) wanted = BLOCK_SIZE;
  size_t received = mic.readBytes((char*)rawBlock, wanted * sizeof(int32_t));
  if (!received || received % sizeof(int32_t)) {
    Serial.println("ERROR: I2S read failed; restarting current window");
    resetCapture();
    return;
  }
  size_t frames = received / sizeof(int32_t);
  for (size_t i = 0; i < frames; ++i) {
    int32_t sample = rawBlock[i] >> 16;
    float adjusted = sample * MIC_GAIN;
    if (adjusted > 32767.0f) adjusted = 32767.0f;
    if (adjusted < -32768.0f) adjusted = -32768.0f;
    int16_t pcm = (int16_t)adjusted;
    pcmBuffer[sampleIndex++] = pcm;
    dbSum += pcm;
    dbSquares += (double)pcm * pcm;
    dbSamples++;
    if (dbSamples == DB_WINDOW) updateDbWindow();
  }
}

void setup() {
  Serial.begin(115200);
  delay(1200);
  Serial.println("\n========= NAS V2 NEW AI 1.0.2 + EXT1 WAKE =========");
  auto wakeReason = esp_sleep_get_wakeup_cause();
  Serial.printf("DeepSleep=%s | WakeReason=%d\n", ENABLE_DEEP_SLEEP ? "ON" : "OFF",
                (int)wakeReason);
  if (wakeReason == ESP_SLEEP_WAKEUP_EXT1) Serial.println("WAKEUP SUCCESS! Radar GPIO11 (EXT1)");
  else Serial.println("BOOT: power on / reset");
  Serial.println("MIC: BCLK8 WS9 SD10 | OLED: SDA16 SCL15 | RADAR: OUT11");
  Serial.println("DFPlayer: ESP RX2 / TX1 / BUSY12 | Radar UART: ESP RX18 / TX17 (unused)");
  Serial.printf("AI MODEL: %u Hz / %u samples\n", (unsigned)EI_CLASSIFIER_FREQUENCY,
                (unsigned)AUDIO_WINDOW);
  if (EI_CLASSIFIER_FREQUENCY != SAMPLE_RATE || AUDIO_WINDOW != 80000) {
    Serial.println("ERROR: model must use 16000Hz / 80000 samples. Check NAS_Classification_fi_inferencing ZIP!");
    while (true) delay(1000);
  }

  pinMode(LED1, OUTPUT); pinMode(LED2, OUTPUT); pinMode(LED3, OUTPUT);
  ledsOff();
  pinMode(RADAR_OUT, INPUT);
  radarLast = isPresent();
  if (!radarLast) noPersonFrom = millis();
  Serial.printf("Radar initial presence = %d\n", radarLast);

  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(50);
  uint8_t oledAddress = 0;
  for (uint8_t address = 0x3C; address <= 0x3D; ++address) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) { oledAddress = address; break; }
  }
  if (oledAddress != 0) oledReady = oled.begin(SSD1306_SWITCHCAPVCC, oledAddress);
  Serial.printf("OLED: %s (I2C 0x%02X)\n", oledReady ? "OK" : "NOT FOUND", oledAddress);
  displayScreen("BOOT");

  Serial.printf("PSRAM size: %u | free: %u bytes\n", (unsigned)ESP.getPsramSize(),
                (unsigned)ESP.getFreePsram());
  Serial.printf("Audio buffer needs %u bytes\n", (unsigned)(AUDIO_WINDOW * sizeof(int16_t)));
  pcmBuffer = (int16_t*)heap_caps_malloc(AUDIO_WINDOW * sizeof(int16_t),
                                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (pcmBuffer == nullptr) {
    Serial.println("FATAL: PSRAM audio buffer allocation FAILED; check OPI PSRAM board setting");
    displayScreen("PSRAM ERROR");
    while (true) delay(1000);
  }
  Serial.println("Audio PSRAM allocation: OK");

  if (!startMic()) {
    displayScreen("MIC ERROR");
    while (true) delay(1000);
  }
  delay(2000);
  warmupMic();
  Serial.println("Microphone I2S: READY");

  if (DF_BUSY >= 0) pinMode(DF_BUSY, INPUT_PULLUP);
  dfPort.begin(9600, SERIAL_8N1, DF_RX, DF_TX);
  dfReady = dfplayer.begin(dfPort, true, true);
  if (dfReady) {
    dfplayer.volume(DF_VOLUME);
    Serial.println("DFPlayer: READY");
  } else {
    Serial.println("DFPlayer: NOT READY; microphone/AI will continue without speaker");
  }

  preferences.begin("nas-settings", false);
  float cached = preferences.getFloat("criticalDb", DEFAULT_CRITICAL_DB);
  if (isfinite(cached) && cached >= MIN_CRITICAL_DB && cached <= MAX_CRITICAL_DB)
    criticalDb = cached;
  Serial.printf("Critical threshold at boot = %.1f dB (default 85, TB shared attribute critical_db)\n", criticalDb);


  uint64_t cachedVersion = preferences.getULong64("webCfgVer", 0);
  float cachedWebDb = preferences.getFloat("webNoiseDb", DEFAULT_CRITICAL_DB);
  if (cachedVersion != 0 && isfinite(cachedWebDb) &&
      cachedWebDb >= WEB_MIN_NOISE_DB && cachedWebDb <= WEB_MAX_NOISE_DB) {
    criticalDb = cachedWebDb;
    webAppliedVersion = cachedVersion;
    webConfigActive = true;
    Serial.printf("Website: restored noise_limit=%.1f version=%llu\n",
                  criticalDb, (unsigned long long)webAppliedVersion);
  }

  if (wifiSettingsValid()) {
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWiFiTryAt = millis();
    mqtt.setServer(TB_HOST, TB_PORT);
    mqtt.setCallback(mqttCallback);
    mqtt.setBufferSize(384);
    mqtt.setSocketTimeout(1);
    mqtt.setKeepAlive(30);
    Serial.println("WiFi: trying connection (nonblocking)");
  } else {
    Serial.println("WiFi/ThingsBoard skipped: configure credentials to enable");
  }
  lastMqttTryAt = millis() - 15000;
  lastTelemetryAt = millis();
  resetCapture();
  Serial.printf("SYSTEM READY - Critical >=%.1f dB (first 0.5s window)\n", criticalDb);
  Serial.printf("LED GREEN GPIO6 >75 / YELLOW GPIO5 >=80 / RED GPIO4 >=%.1f dB\n", criticalDb);
  Serial.println("ThingsBoard legacy key: SHARED critical_db (80..110)");
  Serial.println("Website priority: SHARED noise_limit (0..140) + config_version");
  Serial.println("Website sync: CLIENT applied_config_version; telemetry sound_type, alert, audio_valid, device_status");
  Serial.println("Radar: deep sleep after 30s ABSENT; EXT1 GPIO11 wake");
  Serial.println("Voice commands disabled; DFPlayer critical alert /mp3/0002.mp3 remains enabled");
  Serial.println("Speech/Music/Noise classifier enabled; no keyword recognition");
  displayScreen("LISTENING");
}

void loop() {
  serviceNetwork();
  monitorRadar();
  serviceVoice();

  uint32_t now = millis();
  if (!voicePlaying && !voiceSettling &&
      now - lastTelemetryAt >= TELEMETRY_MS) {
    lastTelemetryAt = now;
    sendEvery30s();
  }


  if (voicePlaying || voiceSettling) {
    delay(2);
    return;
  }
  if (!micReady) {
    if (startMic()) { warmupMic(); resetCapture(); }
    else delay(500);
    return;
  }


  collectMicChunk();
  processAlerts();
  if (!voicePlaying && sampleIndex >= AUDIO_WINDOW) {
    inferAudio();
    sampleIndex = 0;
  }
}
