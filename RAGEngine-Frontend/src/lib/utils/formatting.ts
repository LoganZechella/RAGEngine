export function formatScoreBar(score: number, maxWidth: number = 10): string {
  if (score === null || score === undefined) {
    return '░'.repeat(maxWidth);
  }
  const filled = Math.round(score * maxWidth);
  return '█'.repeat(filled) + '░'.repeat(maxWidth - filled);
}

export function formatConfidence(value: number | null): string {
  if (value === null || value === undefined) {
    return 'Unknown';
  }
  if (value >= 0.8) return 'High';
  if (value >= 0.6) return 'Moderate';
  if (value >= 0.4) return 'Low';
  return 'Very Low';
}

export function formatTime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
} 