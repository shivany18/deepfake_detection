import { useState } from "react";
import LoginForm from "@/components/auth/LoginForm";
import SignupForm from "@/components/auth/SignupForm";
import heroBackground from "@/assets/hero-bg.jpg";

interface AuthPageProps {
  onLogin: (email: string, password: string) => Promise<void>;
  onSignup: (name: string, email: string, password: string) => Promise<void>;
}

export default function AuthPage({ onLogin, onSignup }: AuthPageProps) {
  const [isLoginMode, setIsLoginMode] = useState(true);

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden"
      style={{
        backgroundImage: `url(${heroBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}
    >
      {/* Background Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-background/95 via-background/90 to-background/95" />
      
      {/* Animated Background Elements */}
      <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      
      {/* Content */}
      <div className="relative z-10 w-full max-w-md">
        {isLoginMode ? (
          <LoginForm
            onSubmit={onLogin}
            onSwitchToSignup={() => setIsLoginMode(false)}
          />
        ) : (
          <SignupForm
            onSubmit={onSignup}
            onSwitchToLogin={() => setIsLoginMode(true)}
          />
        )}
      </div>

      {/* Footer */}
      <div className="absolute bottom-4 left-0 right-0 text-center z-10">
        <p className="text-xs text-muted-foreground">
          Powered by AI • Secure • Reliable
        </p>
      </div>
    </div>
  );
}