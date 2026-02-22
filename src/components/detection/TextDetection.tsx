import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { FileText, Scan, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface TextDetectionProps {
  onAnalyze: (text: string) => Promise<{ score: number; isDeepfake: boolean; confidence: number }>;
}

export default function TextDetection({ onAnalyze }: TextDetectionProps) {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{ score: number; isDeepfake: boolean; confidence: number } | null>(null);
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast({
        title: "No Text Provided",
        description: "Please enter some text to analyze",
        variant: "destructive",
      });
      return;
    }

    if (text.length < 10) {
      toast({
        title: "Text Too Short",
        description: "Please provide at least 10 characters for accurate analysis",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const analysisResult = await onAnalyze(text);
      setResult(analysisResult);
      
      toast({
        title: "Analysis Complete",
        description: `Detection confidence: ${analysisResult.confidence}%`,
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze text. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getResultStatus = () => {
    if (!result) return null;
    
    if (result.confidence < 70) {
      return {
        icon: AlertTriangle,
        text: "Uncertain",
        status: "uncertain" as const,
        color: "text-warning"
      };
    }
    
    return {
      icon: result.isDeepfake ? XCircle : CheckCircle,
      text: result.isDeepfake ? "Deepfake Detected" : "Authentic Content",
      status: result.isDeepfake ? "deepfake" as const : "authentic" as const,
      color: result.isDeepfake ? "text-destructive" : "text-success"
    };
  };

  const resultStatus = getResultStatus();

  return (
    <div className="space-y-6">
      <Card className="glass-card animate-slide-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            Text Detection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Enter text to analyze
            </label>
            <Textarea
              placeholder="Paste or type the text you want to analyze for deepfake content..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="min-h-32 bg-background/50 border-border/50 focus:border-primary resize-none"
              maxLength={5000}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Minimum 10 characters required</span>
              <span>{text.length}/5000</span>
            </div>
          </div>

          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing || text.length < 10}
            className="btn-hero w-full"
          >
            {isAnalyzing ? (
              <>
                <Scan className="w-4 h-4 mr-2 animate-spin" />
                Analyzing Text...
              </>
            ) : (
              <>
                <Scan className="w-4 h-4 mr-2" />
                Analyze Text
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {result && resultStatus && (
        <Card className="glass-card animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <resultStatus.icon className={`w-5 h-5 ${resultStatus.color}`} />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center space-y-4">
              <div className="score-circle w-24 h-24 mx-auto">
                <div className="w-full h-full rounded-full bg-gradient-to-r from-primary to-accent p-1">
                  <div className="w-full h-full rounded-full bg-background flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-foreground">
                        {Math.round(result.score * 100)}%
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Score
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <h3 className={`text-lg font-semibold ${resultStatus.color}`}>
                  {resultStatus.text}
                </h3>
                <p className="text-sm text-muted-foreground">
                  Confidence: {result.confidence}%
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-3 rounded-xl bg-secondary/30">
                <div className="text-lg font-bold text-foreground">
                  {result.isDeepfake ? "High Risk" : "Low Risk"}
                </div>
                <div className="text-xs text-muted-foreground">Risk Level</div>
              </div>
              <div className="p-3 rounded-xl bg-secondary/30">
                <div className="text-lg font-bold text-foreground">
                  {Math.round(result.confidence)}%
                </div>
                <div className="text-xs text-muted-foreground">Accuracy</div>
              </div>
            </div>

            <div className={`p-4 rounded-xl border ${
              resultStatus.status === 'authentic' ? 'status-authentic' : 
              resultStatus.status === 'deepfake' ? 'status-deepfake' : 'status-uncertain'
            }`}>
              <p className="text-sm">
                {result.isDeepfake 
                  ? "This text shows signs of being artificially generated. Consider verifying the source."
                  : "This text appears to be authentic human-generated content."
                }
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}