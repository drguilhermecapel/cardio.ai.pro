/**
 * Voice Assistant Component for ECG Analysis
 * Provides voice-controlled interaction with the ECG analysis system
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Settings } from 'lucide-react';

interface VoiceCommand {
  command: string;
  action: string;
  parameters?: { [key: string]: any };
  confidence: number;
  timestamp: number;
}

interface VoiceResponse {
  text: string;
  audio?: Blob;
  actions?: Array<{
    type: string;
    payload: Record<string, unknown>;
  }>;
}

interface ECGAnalysisData {
  rhythm: string;
  predictions: { [condition: string]: number };
  confidence: number;
  interpretability?: {
    attention_maps: { [lead: string]: number[] };
    clinical_findings: Array<{
      condition: string;
      confidence: number;
      evidence: string[];
      lead_involvement: string[];
    }>;
  };
}

interface VoiceSettings {
  language: string;
  voiceSpeed: number;
  voicePitch: number;
  autoSpeak: boolean;
  continuousListening: boolean;
  wakeWord: string;
  confidenceThreshold: number;
}

interface VoiceAssistantProps {
  ecgData?: ECGAnalysisData;
  onCommand?: (command: VoiceCommand) => void;
  onVisualizationControl?: (action: string, params?: Record<string, unknown>) => void;
  isEnabled?: boolean;
}

interface CommandPattern {
  action: string;
  category: string;
  parameters?: { [key: string]: any };
}

const COMMAND_PATTERNS: { [key: string]: CommandPattern } = {
  'analyze ecg': { action: 'analyze_ecg', category: 'analysis' },
  'start analysis': { action: 'start_analysis', category: 'analysis' },
  'stop analysis': { action: 'stop_analysis', category: 'analysis' },
  'show results': { action: 'show_results', category: 'analysis' },
  'explain findings': { action: 'explain_findings', category: 'analysis' },
  
  'show heart model': { action: 'toggle_heart_model', category: 'visualization' },
  'hide heart model': { action: 'toggle_heart_model', parameters: { show: false }, category: 'visualization' },
  'show waveforms': { action: 'toggle_waveforms', category: 'visualization' },
  'zoom in': { action: 'zoom', parameters: { direction: 'in' }, category: 'visualization' },
  'zoom out': { action: 'zoom', parameters: { direction: 'out' }, category: 'visualization' },
  'rotate view': { action: 'rotate_view', category: 'visualization' },
  'reset view': { action: 'reset_view', category: 'visualization' },
  'enter vr mode': { action: 'enter_vr', category: 'visualization' },
  'exit vr mode': { action: 'exit_vr', category: 'visualization' },
  
  'go to dashboard': { action: 'navigate', parameters: { route: '/dashboard' }, category: 'navigation' },
  'open settings': { action: 'navigate', parameters: { route: '/settings' }, category: 'navigation' },
  'show patient list': { action: 'navigate', parameters: { route: '/patients' }, category: 'navigation' },
  
  'what is the rhythm': { action: 'get_rhythm', category: 'information' },
  'what is the confidence': { action: 'get_confidence', category: 'information' },
  'list findings': { action: 'list_findings', category: 'information' },
  'read recommendations': { action: 'read_recommendations', category: 'information' },
  'help': { action: 'show_help', category: 'information' },
  
  'mute': { action: 'mute_audio', category: 'control' },
  'unmute': { action: 'unmute_audio', category: 'control' },
  'repeat': { action: 'repeat_last', category: 'control' },
  'cancel': { action: 'cancel_action', category: 'control' },
};

class VoiceSynthesis {
  private synth: SpeechSynthesis;
  private voices: SpeechSynthesisVoice[];
  private settings: VoiceSettings;

  constructor(settings: VoiceSettings) {
    this.synth = window.speechSynthesis;
    this.voices = [];
    this.settings = settings;
    this.loadVoices();
  }

  private loadVoices(): void {
    this.voices = this.synth.getVoices();
    if (this.voices.length === 0) {
      this.synth.onvoiceschanged = (): void => {
        this.voices = this.synth.getVoices();
      };
    }
  }

  speak(text: string, options?: { priority?: 'high' | 'normal'; interrupt?: boolean }): void {
    if (!this.settings.autoSpeak) return;

    if (options?.interrupt) {
      this.synth.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    
    const preferredVoice = this.voices.find(voice => 
      voice.lang.startsWith(this.settings.language) && voice.name.includes('Neural')
    ) || this.voices.find(voice => 
      voice.lang.startsWith(this.settings.language)
    ) || this.voices[0];

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.rate = this.settings.voiceSpeed;
    utterance.pitch = this.settings.voicePitch;
    utterance.volume = 0.8;

    this.synth.speak(utterance);
  }

  stop(): void {
    this.synth.cancel();
  }

  updateSettings(settings: Partial<VoiceSettings>): void {
    this.settings = { ...this.settings, ...settings };
  }
}

class VoiceRecognition {
  private recognition: any;
  private isListening: boolean = false;
  private settings: VoiceSettings;
  private onResult: (command: VoiceCommand) => void;
  private onError: (error: string) => void;

  constructor(
    settings: VoiceSettings,
    onResult: (command: VoiceCommand) => void,
    onError: (error: string) => void
  ) {
    this.settings = settings;
    this.onResult = onResult;
    this.onError = onError;
    this.initializeRecognition();
  }

  private initializeRecognition(): void {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      this.onError('Speech recognition not supported in this browser');
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    this.recognition = new SpeechRecognition();

    this.recognition.continuous = this.settings.continuousListening;
    this.recognition.interimResults = true;
    this.recognition.lang = this.settings.language;

    this.recognition.onresult = (event: any): void => {
      const last = event.results.length - 1;
      const transcript = event.results[last][0].transcript.toLowerCase().trim();
      const confidence = event.results[last][0].confidence;

      if (event.results[last].isFinal && confidence >= this.settings.confidenceThreshold) {
        this.processCommand(transcript, confidence);
      }
    };

    this.recognition.onerror = (event: any): void => {
      this.onError(`Speech recognition error: ${event.error}`);
    };

    this.recognition.onend = (): void => {
      this.isListening = false;
      if (this.settings.continuousListening) {
        setTimeout(() => this.start(), 1000);
      }
    };
  }

  private processCommand(transcript: string, confidence: number): VoiceCommand | null {
    if (this.settings.continuousListening && this.settings.wakeWord) {
      if (!transcript.includes(this.settings.wakeWord.toLowerCase())) {
        return null;
      }
      transcript = transcript.replace(this.settings.wakeWord.toLowerCase(), '').trim();
    }

    const matchedPattern = Object.entries(COMMAND_PATTERNS).find(([pattern]) => 
      transcript.includes(pattern) || this.fuzzyMatch(transcript, pattern)
    );

    if (matchedPattern) {
      const [, config] = matchedPattern;
      const voiceCommand: VoiceCommand = {
        command: transcript,
        action: config.action,
        parameters: config.parameters || {},
        confidence,
        timestamp: Date.now()
      };

      this.onResult(voiceCommand);
      return voiceCommand;
    } else {
      return this.handleUnknownCommand(transcript, confidence);
    }
  }

  private fuzzyMatch(input: string, pattern: string): boolean {
    const inputWords = input.split(' ');
    const patternWords = pattern.split(' ');
    
    let matches = 0;
    for (const word of patternWords) {
      if (inputWords.some(inputWord => 
        inputWord.includes(word) || word.includes(inputWord)
      )) {
        matches++;
      }
    }
    
    return matches / patternWords.length >= 0.6; // 60% word match threshold
  }

  private handleUnknownCommand(transcript: string, confidence: number): VoiceCommand {
    const parameters: { [key: string]: unknown } = {};
    
    const numbers = transcript.match(/\d+/g);
    if (numbers) {
      parameters.numbers = numbers.map(n => parseInt(n));
    }

    const leads = ['lead one', 'lead two', 'lead three', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'];
    const mentionedLeads = leads.filter(lead => transcript.includes(lead));
    if (mentionedLeads.length > 0) {
      parameters.leads = mentionedLeads;
    }

    const voiceCommand: VoiceCommand = {
      command: transcript,
      action: 'unknown_command',
      parameters,
      confidence,
      timestamp: Date.now()
    };

    this.onResult(voiceCommand);
    return voiceCommand;
  }

  start(): void {
    if (this.recognition && !this.isListening) {
      this.isListening = true;
      this.recognition.start();
    }
  }

  stop(): void {
    if (this.recognition && this.isListening) {
      this.isListening = false;
      this.recognition.stop();
    }
  }

  updateSettings(settings: Partial<VoiceSettings>): void {
    this.settings = { ...this.settings, ...settings };
    if (this.recognition) {
      this.recognition.continuous = this.settings.continuousListening;
      this.recognition.lang = this.settings.language;
    }
  }

  getIsListening(): boolean {
    return this.isListening;
  }
}

export const VoiceAssistant: React.FC<VoiceAssistantProps> = ({
  ecgData,
  onCommand,
  onVisualizationControl,
  isEnabled = true
}) => {
  const [isListening, setIsListening] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [lastCommand, setLastCommand] = useState<VoiceCommand | null>(null);
  const [lastResponse, setLastResponse] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<VoiceSettings>({
    language: 'en-US',
    voiceSpeed: 1.0,
    voicePitch: 1.0,
    autoSpeak: true,
    continuousListening: false,
    wakeWord: 'cardio',
    confidenceThreshold: 0.7
  });

  const voiceSynthesis = useRef<VoiceSynthesis | null>(null);
  const voiceRecognition = useRef<VoiceRecognition | null>(null);
  const commandHistory = useRef<VoiceCommand[]>([]);

  useEffect(() => {
    if (!isEnabled) return;

    voiceSynthesis.current = new VoiceSynthesis(settings);
    
    voiceRecognition.current = new VoiceRecognition(
      settings,
      handleVoiceCommand,
      handleVoiceError
    );

    return () => {
      voiceSynthesis.current?.stop();
      voiceRecognition.current?.stop();
    };
  }, [isEnabled, settings]);

  const handleVoiceCommand = useCallback((command: VoiceCommand): void => {
    setLastCommand(command);
    setIsProcessing(true);
    commandHistory.current.push(command);

    if (commandHistory.current.length > 50) {
      commandHistory.current = commandHistory.current.slice(-50);
    }

    const response = processCommand(command);
    setLastResponse(response.text);

    if (voiceSynthesis.current && !isMuted) {
      voiceSynthesis.current.speak(response.text, { priority: 'high' });
    }

    if (response.actions) {
      response.actions.forEach(action => {
        if (action.type === 'visualization' && onVisualizationControl) {
          onVisualizationControl(command.action, command.parameters);
        }
      });
    }

    if (onCommand) {
      onCommand(command);
    }

    setIsProcessing(false);
  }, [isMuted, onCommand, onVisualizationControl]);

  const processCommand = useCallback((command: VoiceCommand): VoiceResponse => {
    const { action, parameters } = command;

    switch (action) {
      case 'analyze_ecg':
        return {
          text: 'Starting ECG analysis. Please wait while I process the data.',
          actions: [{ type: 'analysis', payload: { action: 'start' } }]
        };

      case 'get_rhythm':
        if (ecgData?.rhythm) {
          return {
            text: `The detected rhythm is ${ecgData.rhythm} with ${(ecgData.confidence * 100).toFixed(1)}% confidence.`
          };
        }
        return { text: 'No ECG data available for rhythm analysis.' };

      case 'get_confidence':
        if (ecgData?.confidence !== undefined) {
          const confidenceLevel = ecgData.confidence > 0.8 ? 'high' : 
                                 ecgData.confidence > 0.6 ? 'moderate' : 'low';
          return {
            text: `The analysis confidence is ${(ecgData.confidence * 100).toFixed(1)}%, which is ${confidenceLevel}.`
          };
        }
        return { text: 'No confidence data available.' };

      case 'list_findings':
        if (ecgData?.interpretability?.clinical_findings?.length) {
          const findings = ecgData.interpretability.clinical_findings
            .slice(0, 3) // Limit to top 3 findings
            .map(f => `${f.condition} with ${(f.confidence * 100).toFixed(1)}% confidence`)
            .join(', ');
          return {
            text: `Key clinical findings include: ${findings}.`
          };
        }
        return { text: 'No significant clinical findings detected.' };

      case 'read_recommendations':
        if (ecgData?.interpretability?.clinical_findings?.length) {
          const findings = ecgData.interpretability.clinical_findings
            .slice(0, 3)
            .map(f => `${f.condition} (${(f.confidence * 100).toFixed(1)}% confidence)`)
            .join(', ');
          return {
            text: `Clinical findings: ${findings}.`
          };
        }
        return { text: 'No clinical findings available.' };

      case 'toggle_heart_model':
        return {
          text: parameters?.show === false ? 'Hiding heart model.' : 'Showing heart model.',
          actions: [{ type: 'visualization', payload: { action: 'toggle_heart_model', params: parameters } }]
        };

      case 'zoom':
        return {
          text: `Zooming ${parameters?.direction || 'in'}.`,
          actions: [{ type: 'visualization', payload: { action: 'zoom', params: parameters } }]
        };

      case 'enter_vr':
        return {
          text: 'Entering VR mode. Please put on your VR headset.',
          actions: [{ type: 'visualization', payload: { action: 'enter_vr' } }]
        };

      case 'show_help':
        return {
          text: 'Available commands include: analyze ECG, show results, explain findings, show heart model, zoom in, zoom out, enter VR mode, and many more. Say "list commands" for a complete list.'
        };

      case 'repeat_last':
        return { text: lastResponse || 'No previous response to repeat.' };

      case 'unknown_command':
        return {
          text: `I didn't understand "${command.command}". Try saying "help" for available commands.`
        };

      default:
        return {
          text: 'Command processed.',
          actions: [{ type: 'general', payload: { action, params: parameters } }]
        };
    }
  }, [ecgData, lastResponse]);

  const handleVoiceError = useCallback((error: string): void => {
    console.error('Voice recognition error:', error);
    if (voiceSynthesis.current && !isMuted) {
      voiceSynthesis.current.speak('Sorry, I had trouble hearing you. Please try again.');
    }
  }, [isMuted]);

  const toggleListening = useCallback((): void => {
    if (!voiceRecognition.current) return;

    if (isListening) {
      voiceRecognition.current.stop();
      setIsListening(false);
    } else {
      voiceRecognition.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const toggleMute = useCallback((): void => {
    setIsMuted(!isMuted);
    if (!isMuted) {
      voiceSynthesis.current?.stop();
    }
  }, [isMuted]);

  const updateSettings = useCallback((newSettings: Partial<VoiceSettings>): void => {
    const updatedSettings = { ...settings, ...newSettings };
    setSettings(updatedSettings);
    voiceSynthesis.current?.updateSettings(updatedSettings);
    voiceRecognition.current?.updateSettings(updatedSettings);
  }, [settings]);

  if (!isEnabled) {
    return null;
  }

  return (
    <div className="voice-assistant" style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      zIndex: 1000,
      background: 'rgba(0, 0, 0, 0.9)',
      borderRadius: '15px',
      padding: '15px',
      color: 'white',
      minWidth: '300px',
      maxWidth: '400px'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
        <h3 style={{ margin: 0, fontSize: '16px' }}>Voice Assistant</h3>
        <button
          onClick={() => setShowSettings(!showSettings)}
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            cursor: 'pointer',
            padding: '5px'
          }}
        >
          <Settings size={16} />
        </button>
      </div>

      {/* Status */}
      <div style={{ marginBottom: '15px' }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '10px',
          fontSize: '14px'
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: isListening ? '#4CAF50' : '#757575'
          }} />
          {isListening ? 'Listening...' : 'Ready'}
          {isProcessing && <span style={{ color: '#FFC107' }}>Processing...</span>}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
        <button
          onClick={toggleListening}
          style={{
            background: isListening ? '#f44336' : '#4CAF50',
            border: 'none',
            borderRadius: '8px',
            color: 'white',
            padding: '10px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            flex: 1
          }}
        >
          {isListening ? <MicOff size={16} /> : <Mic size={16} />}
          {isListening ? 'Stop' : 'Listen'}
        </button>

        <button
          onClick={toggleMute}
          style={{
            background: isMuted ? '#f44336' : '#2196F3',
            border: 'none',
            borderRadius: '8px',
            color: 'white',
            padding: '10px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center'
          }}
        >
          {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
      </div>

      {/* Last Command */}
      {lastCommand && (
        <div style={{ marginBottom: '10px', fontSize: '12px' }}>
          <div style={{ color: '#4CAF50' }}>
            Last command: "{lastCommand.command}"
          </div>
          <div style={{ color: '#2196F3' }}>
            Confidence: {(lastCommand.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* Last Response */}
      {lastResponse && (
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.1)', 
          padding: '10px', 
          borderRadius: '8px',
          fontSize: '12px',
          marginBottom: '10px'
        }}>
          {lastResponse}
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div style={{
          background: 'rgba(255, 255, 255, 0.1)',
          padding: '15px',
          borderRadius: '8px',
          marginTop: '10px'
        }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>Settings</h4>
          
          <div style={{ marginBottom: '10px' }}>
            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>
              Language:
              <select
                value={settings.language}
                onChange={(e) => updateSettings({ language: e.target.value })}
                style={{
                  marginLeft: '10px',
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: 'none',
                  color: 'white',
                  padding: '2px 5px',
                  borderRadius: '4px'
                }}
              >
                <option value="en-US">English (US)</option>
                <option value="en-GB">English (UK)</option>
                <option value="es-ES">Spanish</option>
                <option value="fr-FR">French</option>
                <option value="de-DE">German</option>
                <option value="pt-BR">Portuguese (Brazil)</option>
              </select>
            </label>
          </div>

          <div style={{ marginBottom: '10px' }}>
            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>
              Voice Speed: {settings.voiceSpeed.toFixed(1)}x
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={settings.voiceSpeed}
                onChange={(e) => updateSettings({ voiceSpeed: parseFloat(e.target.value) })}
                style={{ width: '100%', marginTop: '5px' }}
              />
            </label>
          </div>

          <div style={{ marginBottom: '10px' }}>
            <label style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <input
                type="checkbox"
                checked={settings.autoSpeak}
                onChange={(e) => updateSettings({ autoSpeak: e.target.checked })}
              />
              Auto-speak responses
            </label>
          </div>

          <div style={{ marginBottom: '10px' }}>
            <label style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <input
                type="checkbox"
                checked={settings.continuousListening}
                onChange={(e) => updateSettings({ continuousListening: e.target.checked })}
              />
              Continuous listening
            </label>
          </div>

          {settings.continuousListening && (
            <div style={{ marginBottom: '10px' }}>
              <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>
                Wake word:
                <input
                  type="text"
                  value={settings.wakeWord}
                  onChange={(e) => updateSettings({ wakeWord: e.target.value })}
                  style={{
                    marginLeft: '10px',
                    background: 'rgba(255, 255, 255, 0.2)',
                    border: 'none',
                    color: 'white',
                    padding: '2px 5px',
                    borderRadius: '4px'
                  }}
                />
              </label>
            </div>
          )}
        </div>
      )}

      {/* Quick Commands */}
      <div style={{ fontSize: '10px', color: '#999', marginTop: '10px' }}>
        Try: "analyze ECG", "show results", "explain findings", "enter VR mode"
      </div>
    </div>
  );
};

export default VoiceAssistant;

export type { VoiceAssistantProps };

export type { VoiceCommand, VoiceResponse, VoiceSettings, ECGAnalysisData };
