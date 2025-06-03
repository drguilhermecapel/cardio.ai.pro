# Cardio.AI.Pro Frontend

Advanced ECG Analysis Frontend with 3D Visualization and AR/VR Support

## Features

- **3D ECG Visualization**: Real-time 3D visualization of ECG data with interactive heart models
- **AR/VR Support**: Immersive ECG analysis using WebXR technology
- **Voice Assistant**: Hands-free interaction with voice commands
- **Real-time Updates**: WebSocket-based real-time ECG data streaming
- **Responsive Design**: Works across desktop, tablet, and mobile devices

## Technology Stack

- **React 18** with TypeScript
- **Next.js 14** for server-side rendering and routing
- **Three.js** with React Three Fiber for 3D graphics
- **WebXR** for AR/VR capabilities
- **Web Speech API** for voice recognition
- **TailwindCSS** for styling

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Components

### ECGVisualization3D
Advanced 3D visualization component with:
- Real-time ECG waveform rendering
- Interactive 3D heart model
- Attention map visualization
- Clinical findings display

### VoiceAssistant
Voice-controlled interface supporting:
- ECG analysis commands
- Visualization controls
- Natural language queries
- Multi-language support

### ARVRInterface
Immersive ECG analysis with:
- WebXR VR/AR support
- Hand tracking
- Spatial ECG data visualization
- Immersive clinical findings

## Usage

```tsx
import { ECGDashboard } from '@/components/ECGDashboard';

function App() {
  return (
    <ECGDashboard
      enableVoiceAssistant={true}
      enableRealTimeUpdates={true}
      initialMode="3d"
    />
  );
}
```

## Voice Commands

- "analyze ECG" - Start ECG analysis
- "show heart model" - Display 3D heart model
- "enter VR mode" - Switch to VR visualization
- "zoom in/out" - Control visualization zoom
- "what is the rhythm" - Get rhythm information
- "list findings" - Show clinical findings

## Browser Support

- Chrome 90+ (recommended for WebXR)
- Firefox 88+
- Safari 14+
- Edge 90+

## Development

### Project Structure

```
src/
├── components/          # React components
│   ├── ECGVisualization3D.tsx
│   ├── VoiceAssistant.tsx
│   ├── ARVRInterface.tsx
│   └── ECGDashboard.tsx
├── hooks/              # Custom React hooks
│   └── useECGVisualization.ts
├── types/              # TypeScript definitions
│   └── webxr.d.ts
└── utils/              # Utility functions
```

### Testing

```bash
# Run type checking
npm run type-check

# Run linting
npm run lint

# Run tests
npm test
```

## Performance Optimization

- **Code Splitting**: Automatic route-based code splitting
- **WebGL Optimization**: Efficient 3D rendering with Three.js
- **WebSocket Pooling**: Optimized real-time data connections
- **Lazy Loading**: Components loaded on demand

## Accessibility

- **Keyboard Navigation**: Full keyboard support
- **Screen Reader**: ARIA labels and descriptions
- **Voice Control**: Alternative input method
- **High Contrast**: Accessible color schemes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
