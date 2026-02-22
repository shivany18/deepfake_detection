import { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload, File, X, Scan } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface MediaUploadProps {
  type: "image" | "video";
  onUpload: (file: File) => Promise<{ score: number; isDeepfake: boolean; confidence: number }>;
  acceptedFormats: string[];
  maxSize: number; // in MB
}

export default function MediaUpload({ type, onUpload, acceptedFormats, maxSize }: MediaUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState<{ score: number; isDeepfake: boolean; confidence: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const validateFile = (selectedFile: File): boolean => {
    const fileSizeMB = selectedFile.size / (1024 * 1024);
    
    if (fileSizeMB > maxSize) {
      toast({
        title: "File Too Large",
        description: `File size must be less than ${maxSize}MB`,
        variant: "destructive",
      });
      return false;
    }

    const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase();
    if (!fileExtension || !acceptedFormats.includes(`.${fileExtension}`)) {
      toast({
        title: "Invalid File Format",
        description: `Only ${acceptedFormats.join(', ')} files are supported`,
        variant: "destructive",
      });
      return false;
    }

    return true;
  };

  const handleFileSelect = (selectedFile: File) => {
    if (validateFile(selectedFile)) {
      setFile(selectedFile);
      setResult(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const analysisResult = await onUpload(file);
      setResult(analysisResult);
      setUploadProgress(100);
      
      toast({
        title: "Analysis Complete",
        description: `${type} analyzed successfully`,
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: `Unable to analyze ${type}. Please try again.`,
        variant: "destructive",
      });
    } finally {
      clearInterval(progressInterval);
      setIsAnalyzing(false);
    }
  };

  const removeFile = () => {
    setFile(null);
    setResult(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      <Card className="glass-card animate-slide-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 capitalize">
            <Upload className="w-5 h-5 text-primary" />
            {type} Detection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {!file ? (
            <div
              className={`upload-zone ${isDragging ? 'dragover' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold text-foreground mb-2">
                Upload {type}
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Drag and drop your {type} here, or click to browse
              </p>
              <div className="space-y-1 text-xs text-muted-foreground">
                <p>Supported formats: {acceptedFormats.join(', ')}</p>
                <p>Maximum file size: {maxSize}MB</p>
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptedFormats.join(',')}
                onChange={handleFileInputChange}
                className="hidden"
              />
            </div>
          ) : (
            <div className="space-y-4">
              {/* File Preview */}
              <div className="flex items-center justify-between p-4 rounded-xl bg-secondary/30 border border-border/50">
                <div className="flex items-center gap-3">
                  <File className="w-8 h-8 text-primary" />
                  <div>
                    <p className="font-medium text-foreground">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={removeFile}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>

              {/* Image Preview */}
              {type === "image" && file && (
                <div className="rounded-xl overflow-hidden bg-secondary/30">
                  <img
                    src={URL.createObjectURL(file)}
                    alt="Preview"
                    className="w-full h-48 object-cover"
                  />
                </div>
              )}

              {/* Upload Progress */}
              {isAnalyzing && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-foreground">Analyzing {type}...</span>
                    <span className="text-muted-foreground">{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} className="h-2" />
                </div>
              )}

              {/* Analyze Button */}
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="btn-hero w-full"
              >
                {isAnalyzing ? (
                  <>
                    <Scan className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Scan className="w-4 h-4 mr-2" />
                    Analyze {type}
                  </>
                )}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Display */}
      {result && (
        <Card className="glass-card animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Detection Results
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
                <h3 className={`text-lg font-semibold ${
                  result.isDeepfake ? 'text-destructive' : 'text-success'
                }`}>
                  {result.isDeepfake ? 'Deepfake Detected' : 'Authentic Content'}
                </h3>
                <p className="text-sm text-muted-foreground">
                  Confidence: {result.confidence}%
                </p>
              </div>
            </div>

            <div className={`p-4 rounded-xl border ${
              result.isDeepfake ? 'status-deepfake' : 'status-authentic'
            }`}>
              <p className="text-sm">
                {result.isDeepfake 
                  ? `This ${type} shows signs of being artificially generated or manipulated.`
                  : `This ${type} appears to be authentic and unmodified.`
                }
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}