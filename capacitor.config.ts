import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'app.lovable.6d5802c2067842b5b3d6450f905b8a97',
  appName: 'DeepFake Guardian',
  webDir: 'dist',
  server: {
    url: 'https://6d5802c2-0678-42b5-b3d6-450f905b8a97.lovableproject.com?forceHideBadge=true',
    cleartext: true
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: '#1a1625',
      showSpinner: true,
      spinnerColor: '#8b5cf6'
    }
  }
};

export default config;