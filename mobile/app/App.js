/**
 * EchoWarden — Main App
 * Environmental audio intelligence for the deaf and hard-of-hearing.
 */

import React, { useState, useEffect, useRef } from "react";
import { registerRootComponent } from "expo";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  StatusBar,
  Animated,
} from "react-native";
import { Audio } from "expo-av";
import * as Haptics from "expo-haptics";
import { SafeAreaView } from "react-native-safe-area-context";

const SERVER_IP = "10.229.220.83";
const SERVER_PORT = "8000";
const API_URL = `http://${SERVER_IP}:${SERVER_PORT}`;
const POLL_INTERVAL_MS = 2000;
const RECORDING_DURATION_MS = 1500;

const DIRECTION_ICONS = { left: "◀", center: "●", right: "▶" };

const RECORDING_OPTIONS = {
  android: {
    extension: ".wav",
    outputFormat: Audio.AndroidOutputFormat.DEFAULT,
    audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 128000,
  },
  ios: {
    extension: ".wav",
    outputFormat: Audio.IOSOutputFormat.LINEARPCM,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: {},
};

export default function App() {
  const [isListening, setIsListening] = useState(false);
  const [currentEvent, setCurrentEvent] = useState(null);
  const [eventLog, setEventLog] = useState([]);
  const [serverOnline, setServerOnline] = useState(false);
  const [latency, setLatency] = useState(null);

  const recordingRef = useRef(null);
  const intervalRef = useRef(null);
  const isRecordingRef = useRef(false);
  const dangerAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    checkServer();
    return () => stopListening();
  }, []);

  async function checkServer() {
    try {
      const res = await fetch(`${API_URL}/health`);
      const data = await res.json();
      setServerOnline(data.status === "ok");
    } catch {
      setServerOnline(false);
    }
  }

  async function toggleListening() {
    if (isListening) {
      stopListening();
    } else {
      await startListening();
    }
  }

  async function startListening() {
    const { status } = await Audio.requestPermissionsAsync();
    if (status !== "granted") {
      alert("Microphone permission is required for EchoWarden.");
      return;
    }
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });
    setIsListening(true);
    intervalRef.current = setInterval(async () => {
      await classifySample();
    }, POLL_INTERVAL_MS);
  }

  function stopListening() {
    setIsListening(false);
    isRecordingRef.current = false;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (recordingRef.current) {
      recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
  }

  async function classifySample() {
    if (isRecordingRef.current) return;
    isRecordingRef.current = true;

    let recording = null;
    try {
      recording = new Audio.Recording();
      recordingRef.current = recording;

      await recording.prepareToRecordAsync(RECORDING_OPTIONS);
      await recording.startAsync();
      await sleep(RECORDING_DURATION_MS);
      await recording.stopAndUnloadAsync();

      const uri = recording.getURI();
      recordingRef.current = null;
      recording = null;

      if (!uri) return;

      const formData = new FormData();
      formData.append("file", {
        uri,
        type: "audio/wav",
        name: "sample.wav",
      });

      const res = await fetch(`${API_URL}/classify`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const result = await res.json();
        handleClassificationResult(result);
      }
    } catch (err) {
      console.warn("classifySample error:", err.message);
      if (recording) {
        try { await recording.stopAndUnloadAsync(); } catch {}
      }
      recordingRef.current = null;
    } finally {
      isRecordingRef.current = false;
    }
  }

  function handleClassificationResult(result) {
    setLatency(result.latency_ms);
    setCurrentEvent(result);
    if (result.is_danger) triggerDangerAlert();

    const entry = {
      id: Date.now().toString(),
      label: result.top_label,
      confidence: result.events[0]?.confidence ?? 0,
      is_danger: result.is_danger,
      direction: result.events[0]?.direction ?? "center",
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };
    setEventLog((prev) => [entry, ...prev].slice(0, 30));
  }

  function triggerDangerAlert() {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    Animated.sequence([
      Animated.timing(dangerAnim, { toValue: 1, duration: 80, useNativeDriver: false }),
      Animated.timing(dangerAnim, { toValue: 0, duration: 300, useNativeDriver: false }),
    ]).start();
  }

  const bgColor = dangerAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ["transparent", "rgba(220, 38, 38, 0.25)"],
  });

  const isDanger = currentEvent?.is_danger;
  const topEvent = currentEvent?.events?.[0];

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <Animated.View
        style={[StyleSheet.absoluteFill, { backgroundColor: bgColor }]}
        pointerEvents="none"
      />

      <View style={styles.header}>
        <Text style={styles.title}>EchoWarden</Text>
        <View style={[styles.serverDot, { backgroundColor: serverOnline ? "#16a34a" : "#dc2626" }]} />
        <TouchableOpacity onPress={checkServer} style={styles.refreshBtn}>
          <Text style={styles.refreshText}>⟳</Text>
        </TouchableOpacity>
      </View>

      <View style={[styles.mainCard, isDanger && styles.mainCardDanger]}>
        {currentEvent ? (
          <>
            <Text style={styles.directionIcon}>
              {DIRECTION_ICONS[topEvent?.direction ?? "center"]}
            </Text>
            <Text style={[styles.soundLabel, isDanger && styles.soundLabelDanger]}>
              {currentEvent.top_label}
            </Text>
            <View style={styles.confidenceBar}>
              <View
                style={[
                  styles.confidenceFill,
                  {
                    width: `${Math.round((topEvent?.confidence ?? 0) * 100)}%`,
                    backgroundColor: isDanger ? "#dc2626" : "#2563eb",
                  },
                ]}
              />
            </View>
            <Text style={styles.confidenceText}>
              {Math.round((topEvent?.confidence ?? 0) * 100)}% confidence
              {latency ? `  ·  ${Math.round(latency)}ms` : ""}
            </Text>
            {isDanger && (
              <View style={styles.dangerBadge}>
                <Text style={styles.dangerBadgeText}>⚠ DANGER SOUND</Text>
              </View>
            )}
          </>
        ) : (
          <Text style={styles.idleText}>
            {isListening ? "Listening…" : "Tap Start to begin"}
          </Text>
        )}
      </View>

      <TouchableOpacity
        style={[styles.startBtn, isListening && styles.stopBtn]}
        onPress={toggleListening}
        activeOpacity={0.8}
      >
        <Text style={styles.startBtnText}>
          {isListening ? "Stop Listening" : "Start Listening"}
        </Text>
      </TouchableOpacity>

      {!serverOnline && (
        <Text style={styles.serverWarning}>
          ⚠ Cannot reach server at {API_URL}{"\n"}
          Edit SERVER_IP in App.js to your laptop's local IP.
        </Text>
      )}

      <Text style={styles.logTitle}>Recent sounds</Text>
      <FlatList
        data={eventLog}
        keyExtractor={(item) => item.id}
        style={styles.logList}
        renderItem={({ item }) => (
          <View style={[styles.logRow, item.is_danger && styles.logRowDanger]}>
            <Text style={styles.logDir}>{DIRECTION_ICONS[item.direction]}</Text>
            <Text style={[styles.logLabel, item.is_danger && styles.logLabelDanger]}>
              {item.label}
            </Text>
            <Text style={styles.logConf}>{Math.round(item.confidence * 100)}%</Text>
            <Text style={styles.logTime}>{item.time}</Text>
          </View>
        )}
        ListEmptyComponent={
          <Text style={styles.emptyLog}>No sounds detected yet.</Text>
        }
      />
    </SafeAreaView>
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8f8f7", paddingHorizontal: 16 },
  header: { flexDirection: "row", alignItems: "center", paddingVertical: 14 },
  title: { fontSize: 20, fontWeight: "600", color: "#1a1a1a", flex: 1 },
  serverDot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  refreshBtn: { padding: 4 },
  refreshText: { fontSize: 18, color: "#888" },
  mainCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    borderWidth: 0.5,
    borderColor: "#e0dfd8",
    padding: 24,
    alignItems: "center",
    minHeight: 180,
    justifyContent: "center",
    marginBottom: 16,
  },
  mainCardDanger: { borderColor: "#dc2626", borderWidth: 1.5, backgroundColor: "#fef2f2" },
  directionIcon: { fontSize: 28, color: "#555", marginBottom: 8 },
  soundLabel: { fontSize: 26, fontWeight: "600", color: "#1a1a1a", textAlign: "center", marginBottom: 12 },
  soundLabelDanger: { color: "#dc2626" },
  confidenceBar: { width: "100%", height: 6, backgroundColor: "#f0efea", borderRadius: 3, marginBottom: 6 },
  confidenceFill: { height: 6, borderRadius: 3 },
  confidenceText: { fontSize: 12, color: "#888" },
  dangerBadge: {
    marginTop: 12, backgroundColor: "#dc2626",
    paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20,
  },
  dangerBadgeText: { color: "#fff", fontWeight: "600", fontSize: 13, letterSpacing: 0.5 },
  idleText: { fontSize: 16, color: "#aaa", fontStyle: "italic" },
  startBtn: {
    backgroundColor: "#1a1a1a", borderRadius: 12, paddingVertical: 16,
    alignItems: "center", marginBottom: 12,
  },
  stopBtn: { backgroundColor: "#dc2626" },
  startBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  serverWarning: { fontSize: 12, color: "#dc2626", textAlign: "center", marginBottom: 12, lineHeight: 18 },
  logTitle: { fontSize: 13, fontWeight: "500", color: "#888", marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 },
  logList: { flex: 1 },
  logRow: {
    flexDirection: "row", alignItems: "center",
    paddingVertical: 10, borderBottomWidth: 0.5, borderBottomColor: "#ebe9e0",
  },
  logRowDanger: { backgroundColor: "#fff5f5" },
  logDir: { fontSize: 14, color: "#888", width: 24 },
  logLabel: { flex: 1, fontSize: 14, color: "#1a1a1a" },
  logLabelDanger: { color: "#dc2626", fontWeight: "500" },
  logConf: { fontSize: 12, color: "#aaa", width: 38, textAlign: "right" },
  logTime: { fontSize: 11, color: "#bbb", width: 70, textAlign: "right" },
  emptyLog: { fontSize: 13, color: "#ccc", textAlign: "center", paddingTop: 20, fontStyle: "italic" },
});

registerRootComponent(App);