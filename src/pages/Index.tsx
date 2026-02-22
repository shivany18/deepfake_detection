import { useState, useEffect } from "react";
import AuthPage from "./AuthPage";
import MainApp from "./MainApp";
import { useToast } from "@/hooks/use-toast";

interface User {
  id: string;
  name: string;
  email: string;
}

const Index = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  // Check for existing session on mount
  useEffect(() => {
    const checkSession = () => {
      const savedUser = localStorage.getItem('deepfake_user');
      if (savedUser) {
        try {
          setUser(JSON.parse(savedUser));
        } catch (error) {
          localStorage.removeItem('deepfake_user');
        }
      }
      setIsLoading(false);
    };

    checkSession();
  }, []);

  const handleLogin = async (email: string, password: string) => {
    setIsLoading(true);
    
    // Mock API call - replace with actual authentication
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Simulate authentication logic
    if (email === "demo@example.com" && password === "password") {
      const mockUser: User = {
        id: "1",
        name: "Demo User",
        email: email,
      };
      
      setUser(mockUser);
      localStorage.setItem('deepfake_user', JSON.stringify(mockUser));
      
      toast({
        title: "Welcome back!",
        description: "Successfully logged in to DeepFake Guardian",
      });
    } else {
      throw new Error("Invalid credentials");
    }
    
    setIsLoading(false);
  };

  const handleSignup = async (name: string, email: string, password: string) => {
    setIsLoading(true);
    
    // Mock API call - replace with actual registration
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const newUser: User = {
      id: Date.now().toString(),
      name: name,
      email: email,
    };
    
    setUser(newUser);
    localStorage.setItem('deepfake_user', JSON.stringify(newUser));
    
    toast({
      title: "Account Created!",
      description: "Welcome to DeepFake Guardian",
    });
    
    setIsLoading(false);
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('deepfake_user');
    
    toast({
      title: "Logged Out",
      description: "See you next time!",
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 mx-auto bg-gradient-to-br from-primary to-accent rounded-2xl flex items-center justify-center animate-pulse-glow">
            <div className="w-8 h-8 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
          </div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            DeepFake Guardian
          </h1>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <AuthPage 
        onLogin={handleLogin}
        onSignup={handleSignup}
      />
    );
  }

  return <MainApp user={user} onLogout={handleLogout} />;
};

export default Index;
