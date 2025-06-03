/**
 * WebXR API type definitions
 */

declare global {
  interface Navigator {
    xr?: XRSystem;
  }

  interface XRSystem {
    isSessionSupported(mode: XRSessionMode): Promise<boolean>;
    requestSession(mode: XRSessionMode, options?: XRSessionInit): Promise<XRSession>;
  }

  interface XRSession extends EventTarget {
    mode: XRSessionMode;
    addEventListener(type: 'end', listener: () => void): void;
    end(): Promise<void>;
  }

  type XRSessionMode = 'inline' | 'immersive-vr' | 'immersive-ar';

  interface XRSessionInit {
    requiredFeatures?: string[];
    optionalFeatures?: string[];
  }
}

export {};
