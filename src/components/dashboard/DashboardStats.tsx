import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Image, Video, Shield, TrendingUp, Clock } from "lucide-react";

interface DashboardStatsProps {
  stats: {
    totalScans: number;
    textScans: number;
    imageScans: number;
    videoScans: number;
    detectedFakes: number;
    accuracy: number;
  };
}

export default function DashboardStats({ stats }: DashboardStatsProps) {
  const statCards = [
    {
      title: "Total Scans",
      value: stats.totalScans,
      icon: Shield,
      color: "text-primary",
      trend: "+12%"
    },
    {
      title: "Text Analysis",
      value: stats.textScans,
      icon: FileText,
      color: "text-accent",
      trend: "+5%"
    },
    {
      title: "Image Detection",
      value: stats.imageScans,
      icon: Image,
      color: "text-success",
      trend: "+8%"
    },
    {
      title: "Video Analysis",
      value: stats.videoScans,
      icon: Video,
      color: "text-warning",
      trend: "+15%"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        {statCards.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.title} className="glass-card animate-slide-up">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-muted-foreground font-medium">
                      {stat.title}
                    </p>
                    <p className="text-2xl font-bold text-foreground">
                      {stat.value}
                    </p>
                    <p className="text-xs text-success flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      {stat.trend}
                    </p>
                  </div>
                  <div className={`w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center`}>
                    <Icon className={`w-5 h-5 ${stat.color}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Detection Summary */}
      <Card className="glass-card animate-slide-up">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            Detection Summary
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Detected Deepfakes</span>
            <span className="text-lg font-bold text-destructive">{stats.detectedFakes}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Detection Accuracy</span>
            <span className="text-lg font-bold text-success">{stats.accuracy}%</span>
          </div>

          {/* Accuracy Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Accuracy Rate</span>
              <span>{stats.accuracy}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div 
                className="progress-gradient h-2 rounded-full transition-all duration-500"
                style={{ width: `${stats.accuracy}%` }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card className="glass-card animate-slide-up">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            Recent Activity
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[
            { type: "Image", result: "Authentic", time: "2 min ago", status: "authentic" },
            { type: "Text", result: "Deepfake", time: "15 min ago", status: "deepfake" },
            { type: "Video", result: "Processing", time: "32 min ago", status: "uncertain" }
          ].map((activity, index) => (
            <div key={index} className="flex items-center justify-between p-3 rounded-xl bg-secondary/30">
              <div className="flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${
                  activity.status === 'authentic' ? 'bg-success' : 
                  activity.status === 'deepfake' ? 'bg-destructive' : 'bg-warning'
                }`} />
                <div>
                  <p className="text-sm font-medium text-foreground">{activity.type} Detection</p>
                  <p className="text-xs text-muted-foreground">{activity.time}</p>
                </div>
              </div>
              <span className={`text-xs px-2 py-1 rounded-full border ${
                activity.status === 'authentic' ? 'status-authentic' : 
                activity.status === 'deepfake' ? 'status-deepfake' : 'status-uncertain'
              }`}>
                {activity.result}
              </span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}