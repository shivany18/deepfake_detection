import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FileText, Image, Video, Calendar, Eye, Download } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

interface HistoryItem {
  id: string;
  type: "text" | "image" | "video";
  fileName?: string;
  content?: string;
  result: {
    score: number;
    isDeepfake: boolean;
    confidence: number;
  };
  timestamp: Date;
  fileUrl?: string;
}

interface HistoryListProps {
  items: HistoryItem[];
  onViewDetails: (item: HistoryItem) => void;
}

export default function HistoryList({ items, onViewDetails }: HistoryListProps) {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case "text":
        return FileText;
      case "image":
        return Image;
      case "video":
        return Video;
      default:
        return FileText;
    }
  };

  const getStatusBadge = (result: HistoryItem["result"]) => {
    if (result.confidence < 70) {
      return (
        <Badge variant="outline" className="status-uncertain">
          Uncertain
        </Badge>
      );
    }
    
    return (
      <Badge variant="outline" className={result.isDeepfake ? "status-deepfake" : "status-authentic"}>
        {result.isDeepfake ? "Deepfake" : "Authentic"}
      </Badge>
    );
  };

  if (items.length === 0) {
    return (
      <Card className="glass-card">
        <CardContent className="text-center py-12">
          <Calendar className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-lg font-semibold text-foreground mb-2">No History Yet</h3>
          <p className="text-muted-foreground">
            Your detection history will appear here after you analyze content.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-primary" />
            Detection History
          </CardTitle>
        </CardHeader>
      </Card>

      {items.map((item) => {
        const Icon = getTypeIcon(item.type);
        
        return (
          <Card key={item.id} className="glass-card animate-slide-up">
            <CardContent className="p-4">
              <div className="space-y-3">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                      <Icon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium text-foreground capitalize">
                        {item.type} Detection
                      </h4>
                      <p className="text-xs text-muted-foreground">
                        {formatDistanceToNow(item.timestamp, { addSuffix: true })}
                      </p>
                    </div>
                  </div>
                  {getStatusBadge(item.result)}
                </div>

                {/* Content Preview */}
                <div className="space-y-2">
                  {item.fileName && (
                    <p className="text-sm text-foreground font-medium">
                      📎 {item.fileName}
                    </p>
                  )}
                  
                  {item.content && (
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {item.content.length > 100 
                        ? `${item.content.substring(0, 100)}...` 
                        : item.content
                      }
                    </p>
                  )}

                  {item.type === "image" && item.fileUrl && (
                    <div className="w-full h-24 rounded-lg bg-secondary/30 overflow-hidden">
                      <img
                        src={item.fileUrl}
                        alt="Analyzed content"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}
                </div>

                {/* Results Summary */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-secondary/30">
                  <div className="grid grid-cols-2 gap-4 flex-1">
                    <div className="text-center">
                      <div className="text-lg font-bold text-foreground">
                        {Math.round(item.result.score * 100)}%
                      </div>
                      <div className="text-xs text-muted-foreground">Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-foreground">
                        {item.result.confidence}%
                      </div>
                      <div className="text-xs text-muted-foreground">Confidence</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 ml-4">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onViewDetails(item)}
                      className="text-muted-foreground hover:text-primary"
                    >
                      <Eye className="w-4 h-4" />
                    </Button>
                    {item.fileUrl && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => {
                          const link = document.createElement('a');
                          link.href = item.fileUrl!;
                          link.download = item.fileName || 'download';
                          link.click();
                        }}
                        className="text-muted-foreground hover:text-primary"
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}