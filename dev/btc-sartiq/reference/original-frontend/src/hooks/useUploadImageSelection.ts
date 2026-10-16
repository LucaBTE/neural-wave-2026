import { useCallback, useEffect, useRef, useState } from "react";

export function useUploadImageSelection() {
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadPreviewUrls, setUploadPreviewUrls] = useState<string[]>([]);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const clearUploadFiles = useCallback(() => {
    setUploadPreviewUrls((prev) => {
      prev.forEach((previewUrl) => URL.revokeObjectURL(previewUrl));
      return [];
    });
    setUploadFiles([]);

    if (uploadInputRef.current) {
      uploadInputRef.current.value = "";
    }
  }, []);

  const onSelectUploadFiles = useCallback((inputFiles: FileList | null) => {
    const nextFiles = Array.from(inputFiles ?? []).filter((file) => file.type.startsWith("image/"));
    if (nextFiles.length === 0) {
      return;
    }

    setUploadPreviewUrls((prev) => {
      prev.forEach((previewUrl) => URL.revokeObjectURL(previewUrl));
      return nextFiles.map((file) => URL.createObjectURL(file));
    });
    setUploadFiles(nextFiles);
  }, []);

  useEffect(() => {
    return () => {
      uploadPreviewUrls.forEach((previewUrl) => URL.revokeObjectURL(previewUrl));
    };
  }, [uploadPreviewUrls]);

  return {
    uploadFiles,
    uploadPreviewUrls,
    uploadInputRef,
    clearUploadFiles,
    onSelectUploadFiles,
  };
}
