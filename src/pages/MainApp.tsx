import { useState, useEffect } from "react";
import Header from "@/components/layout/Header";
import MobileNav from "@/components/layout/MobileNav";
import DashboardStats from "@/components/dashboard/DashboardStats";
import TextDetection from "@/components/detection/TextDetection";
import MediaUpload from "@/components/detection/MediaUpload";
import HistoryList from "@/components/history/HistoryList";
import { useToast } from "@/hooks/use-toast";

interface User {
  id: string;
  name: string;
  email: string;
}

interface MainAppProps {
  user: User;
  onLogout: () => void;
}

// Mock API functions - Replace with actual API calls
const mockAnalyzeText = async (text: string) => {
  await new Promise(resolve => setTimeout(resolve, 2000));
  const score = Math.random();
  const isDeepfake = score > 0.7;
  const confidence = Math.floor(Math.random() * 30) + 70;
  return { score, isDeepfake, confidence };
};

const mockAnalyzeMedia = async (file: File) => {
  await new Promise(resolve => setTimeout(resolve, 3000));
  const score = Math.random();
  const isDeepfake = score > 0.6;
  const confidence = Math.floor(Math.random() * 25) + 75;
  return { score, isDeepfake, confidence };
};

export default function MainApp({ user, onLogout }: MainAppProps) {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [stats, setStats] = useState({
    totalScans: 0,
    textScans: 0,
    imageScans: 0,
    videoScans: 0,
    detectedFakes: 0,
    accuracy: 94,
  });
  const [history, setHistory] = useState<any[]>([]);
  const { toast } = useToast();

  // Load user data on mount
  useEffect(() => {
    // Mock loading user stats
    const loadUserStats = () => {
      setStats({
        totalScans: 127,
        textScans: 45,
        imageScans: 52,
        videoScans: 30,
        detectedFakes: 23,
        accuracy: 94,
      });

      // Mock history data
      const mockHistory = [
        {
          id: "1",
          type: "text",
          content: "This is a sample text that was analyzed for deepfake detection...",
          result: { score: 0.85, isDeepfake: true, confidence: 89 },
          timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
        },
        {
          id: "2",
          type: "image",
          fileName: "profile_photo.jpg",
          result: { score: 0.25, isDeepfake: false, confidence: 92 },
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
          fileUrl: "/placeholder.svg",
        },
        {
          id: "3",
          type: "video",
          fileName: "speech_video.mp4",
          result: { score: 0.65, isDeepfake: false, confidence: 78 },
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
        },
      ];
      setHistory(mockHistory);
    };

    loadUserStats();
  }, []);

  const handleTextAnalyze = async (text: string) => {
    const result = await mockAnalyzeText(text);
    
    // Add to history
    const newHistoryItem = {
      id: Date.now().toString(),
      type: "text",
      content: text,
      result,
      timestamp: new Date(),
    };
    setHistory(prev => [newHistoryItem, ...prev]);
    
    // Update stats
    setStats(prev => ({
      ...prev,
      totalScans: prev.totalScans + 1,
      textScans: prev.textScans + 1,
      detectedFakes: result.isDeepfake ? prev.detectedFakes + 1 : prev.detectedFakes,
    }));

    return result;
  };

  const handleMediaAnalyze = async (file: File) => {
    const result = await mockAnalyzeMedia(file);
    const fileUrl = URL.createObjectURL(file);
    
    // Add to history
    const newHistoryItem = {
      id: Date.now().toString(),
      type: file.type.startsWith('image/') ? "image" : "video",
      fileName: file.name,
      result,
      timestamp: new Date(),
      fileUrl,
    };
    setHistory(prev => [newHistoryItem, ...prev]);
    
    // Update stats
    const isImage = file.type.startsWith('image/');
    setStats(prev => ({
      ...prev,
      totalScans: prev.totalScans + 1,
      imageScans: isImage ? prev.imageScans + 1 : prev.imageScans,
      videoScans: !isImage ? prev.videoScans + 1 : prev.videoScans,
      detectedFakes: result.isDeepfake ? prev.detectedFakes + 1 : prev.detectedFakes,
    }));

    return result;
  };

  const handleViewDetails = (item: any) => {
    toast({
      title: "View Details",
      description: `Viewing details for ${item.type} detection`,
    });
  };

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":
        return <DashboardStats stats={stats} />;
      
      case "text":
        return <TextDetection onAnalyze={handleTextAnalyze} />;
      
      case "image":
        return (
          <MediaUpload
            type="image"
            onUpload={handleMediaAnalyze}
            acceptedFormats={['.jpg', '.jpeg', '.png', '.webp']}
            maxSize={10}
          />
        );
      
      case "video":
        return (
          <MediaUpload
            type="video"
            onUpload={handleMediaAnalyze}
            acceptedFormats={['.mp4', '.avi', '.mov', '.webm']}
            maxSize={100}
          />
        );
      
      case "history":
        return (
          <HistoryList
            items={history}
            onViewDetails={handleViewDetails}
          />
        );
      
      default:
        return <DashboardStats stats={stats} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header userName={user.name} onLogout={onLogout} />
      
      <main className="pt-20 pb-20 px-4">
        <div className="max-w-md mx-auto">
          {renderContent()}
        </div>
      </main>
      
      <MobileNav activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  );
}