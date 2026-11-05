/**
 * API Client: Real-time bridge connecting the 3D frontend to the FastAPI backend.
 * Powers live SSE telemetry streaming, conformal RUL inference, and autonomous MRO strategy.
 */

const API_BASE = 'http://localhost:8000';

class BackendAPIClient {
  constructor() {
    this.baseUrl = API_BASE;
    this.isConnected = false;
    this.activeEventSource = null;
    this.modelInfo = null;
    this.statusListeners = new Set();
  }

  onStatusChange(fn) {
    this.statusListeners.add(fn);
    return () => this.statusListeners.delete(fn);
  }

  notifyStatus(connected, data = null) {
    this.isConnected = connected;
    for (const fn of this.statusListeners) {
      fn({ connected, data, modelInfo: this.modelInfo });
    }
  }

  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        await this.fetchInfo();
        this.notifyStatus(true, data);
        return true;
      }
      this.notifyStatus(false);
      return false;
    } catch {
      this.notifyStatus(false);
      return false;
    }
  }

  async fetchInfo() {
    try {
      const res = await fetch(`${this.baseUrl}/`, { method: 'GET' });
      if (res.ok) {
        this.modelInfo = await res.json();
        return this.modelInfo;
      }
    } catch {
      // Offline fallback
    }
    return null;
  }

  /**
   * Start live SSE telemetry streaming for an engine / satellite subsystem.
   */
  startStream(unitId = 1, onMessage, onError) {
    this.stopStream();

    const streamUrl = `${this.baseUrl}/stream/${unitId}?delay=0.15`;
    this.activeEventSource = new EventSource(streamUrl);

    this.activeEventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (onMessage) onMessage(payload);
      } catch (err) {
        console.error('Failed to parse SSE telemetry packet:', err);
      }
    };

    this.activeEventSource.onerror = (err) => {
      console.warn('SSE telemetry stream disconnected or closed:', err);
      if (onError) onError(err);
      this.stopStream();
    };

    return this.activeEventSource;
  }

  stopStream() {
    if (this.activeEventSource) {
      this.activeEventSource.close();
      this.activeEventSource = null;
    }
  }

  /**
   * Request automated maintenance strategy and simulated parts order from AI agent.
   */
  async requestMaintenanceStrategy(unitId = 1, predictedRul = 25.0) {
    try {
      const res = await fetch(`${this.baseUrl}/maintenance-strategy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          unit_id: unitId,
          predicted_rul: predictedRul,
          dominant_sensor: 'sensor_11',
          operating_cycle: 180,
        }),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (err) {
      console.error('Maintenance strategy API call failed:', err);
    }
    return null;
  }
}

export const apiClient = new BackendAPIClient();
