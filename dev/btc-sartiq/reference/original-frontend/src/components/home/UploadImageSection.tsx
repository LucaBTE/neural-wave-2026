import type { ChangeEvent, DragEvent, FormEvent, RefObject } from "react";

import { Box, Button, HStack, Stack, Text, VStack, useColorModeValue } from "@chakra-ui/react";
import { BiCloudUpload, BiFolderOpen, BiTrash, BiUpload } from "react-icons/bi";

import { GlassPanel } from "../common/GlassPanel";

interface UploadProjectOption {
  id: string;
  name: string;
}

interface UploadImageSectionProps {
  projects: UploadProjectOption[];
  uploadFiles: File[];
  uploadPreviewUrls: string[];
  uploadProjectId: string;
  isDragOverUpload: boolean;
  submitting: boolean;
  uploadInputRef: RefObject<HTMLInputElement>;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onInputChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onUploadProjectChange: (value: string) => void;
  onOpenFilePicker: () => void;
  onClearFiles: () => void;
  onDropUpload: (event: DragEvent<HTMLDivElement>) => void;
  onDragOverUpload: (event: DragEvent<HTMLDivElement>) => void;
  onDragLeaveUpload: (event: DragEvent<HTMLDivElement>) => void;
}

export function UploadImageSection({
  projects,
  uploadFiles,
  uploadPreviewUrls,
  uploadProjectId,
  isDragOverUpload,
  submitting,
  uploadInputRef,
  onSubmit,
  onInputChange,
  onUploadProjectChange,
  onOpenFilePicker,
  onClearFiles,
  onDropUpload,
  onDragOverUpload,
  onDragLeaveUpload,
}: UploadImageSectionProps) {
  const mutedText = useColorModeValue("rgba(245,241,235,0.72)", "whiteAlpha.700");
  const dropBorderColor = useColorModeValue(
    isDragOverUpload ? "rgba(216,194,164,0.28)" : "rgba(255,255,255,0.14)",
    isDragOverUpload ? "rgba(240,246,252,0.24)" : "rgba(240,246,252,0.16)",
  );
  const dropBg = useColorModeValue(
    isDragOverUpload ? "rgba(255,255,255,0.05)" : "transparent",
    isDragOverUpload ? "#1b2430" : "#151b23",
  );
  const previewPanelBg = useColorModeValue("transparent", "#1b2430");
  const previewPanelBorder = useColorModeValue("rgba(255,255,255,0.12)", "rgba(240,246,252,0.12)");
  const sidebarBg = useColorModeValue("transparent", "#1b2430");
  const sidebarBorder = useColorModeValue("rgba(255,255,255,0.12)", "rgba(240,246,252,0.12)");
  const previewStripBg = useColorModeValue("transparent", "rgba(11,18,28,0.58)");
  const projectListBg = useColorModeValue("transparent", "rgba(11,18,28,0.42)");
  const projectListBorder = useColorModeValue("rgba(255,255,255,0.08)", "rgba(240,246,252,0.08)");
  const projectButtonHoverBg = useColorModeValue("rgba(255,255,255,0.04)", "rgba(255,255,255,0.08)");
  const selectedProjectBg = useColorModeValue("#20262d", "rgba(96,165,250,0.22)");
  const selectedProjectBorder = useColorModeValue("rgba(255,255,255,0.08)", "rgba(147,197,253,0.44)");
  const selectedProjectShadow = useColorModeValue("inset 0 0 0 1px rgba(255,255,255,0.04)", "none");

  return (
    <GlassPanel p={{ base: 4, md: 5 }} lightBg="transparent" darkBg="#151b23">
      <Text fontWeight="800" fontSize={{ base: "xl", md: "2xl" }} mb={1} letterSpacing="-0.04em">
        Upload
      </Text>
      <Text color={mutedText} mb={5}>
        Select a project and add images.
      </Text>

      <form onSubmit={onSubmit}>
        <Stack direction={{ base: "column", xl: "row" }} align="stretch" spacing={5}>
          <VStack align="stretch" spacing={4} flex="1" minW={0}>
            <input
              ref={uploadInputRef}
              type="file"
              multiple
              accept="image/png,image/jpg,image/jpeg,image/webp,image/bmp"
              style={{ display: "none" }}
              onChange={onInputChange}
              disabled={projects.length === 0}
            />

            {uploadFiles.length === 0 ? (
              <Box
                border="1px dashed"
                borderColor={dropBorderColor}
                bg={dropBg}
                borderRadius="8px"
                px={5}
                py={10}
                textAlign="center"
                onDrop={onDropUpload}
                onDragOver={onDragOverUpload}
                onDragLeave={onDragLeaveUpload}
              >
                <BiCloudUpload
                  size={52}
                  style={{ display: "block", margin: "0 auto 14px auto", opacity: 0.92 }}
                  color="currentColor"
                />
                <Text fontSize="lg" fontWeight="700" mb={2}>
                  Drop files here
                </Text>
                <Text color={mutedText} mb={5}>
                  PNG, JPG, JPEG, WEBP, BMP
                </Text>
                <Button
                  size="sm"
                  leftIcon={<BiFolderOpen />}
                  onClick={onOpenFilePicker}
                  isDisabled={projects.length === 0}
                >
                  Choose files
                </Button>
              </Box>
            ) : (
              <Box flex="1" minW={0}>
                <Text
                  fontSize="xs"
                  color={mutedText}
                  mb={2}
                  textTransform="uppercase"
                  letterSpacing="0.12em"
                >
                  Preview
                </Text>
                <Box
                  position="relative"
                  h={{ base: "220px", md: "280px" }}
                  w="full"
                  borderRadius="8px"
                  overflow="hidden"
                  border="1px solid"
                  borderColor={previewPanelBorder}
                  bg={previewPanelBg}
                >
                  <img
                    src={uploadPreviewUrls[0]}
                    alt={uploadFiles[0]?.name || "Upload preview"}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      objectPosition: "center",
                      display: "block",
                      background: "transparent",
                    }}
                  />
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    position="absolute"
                    top={3}
                    right={3}
                    leftIcon={<BiTrash />}
                    onClick={onClearFiles}
                  >
                    Clear
                  </Button>
                </Box>
                {uploadPreviewUrls.length > 1 ? (
                  <Box
                    mt={3}
                    p={2}
                    borderRadius="8px"
                    border="1px solid"
                    borderColor={previewPanelBorder}
                    bg={previewStripBg}
                    overflow="hidden"
                  >
                    <HStack
                      gap={2}
                      overflowX="auto"
                      py={1}
                      pr={1}
                      minW={0}
                      sx={{
                        "&::-webkit-scrollbar": { height: "8px" },
                        "&::-webkit-scrollbar-thumb": {
                          background: previewPanelBorder,
                          borderRadius: "999px",
                        },
                        "&::-webkit-scrollbar-track": {
                          background: "transparent",
                        },
                      }}
                    >
                    {uploadPreviewUrls.slice(1).map((previewUrl, index) => (
                      <Box
                        key={previewUrl}
                        w="68px"
                        h="68px"
                        borderRadius="6px"
                        overflow="hidden"
                        border="1px solid"
                        borderColor={previewPanelBorder}
                        flexShrink={0}
                      >
                        <img
                          src={previewUrl}
                          alt={uploadFiles[index + 1]?.name || `Preview ${index + 2}`}
                          style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
                        />
                      </Box>
                    ))}
                    </HStack>
                  </Box>
                ) : null}
                <Text
                  fontSize="sm"
                  color={mutedText}
                  mt={3}
                  noOfLines={1}
                >
                  {uploadFiles.length === 1 ? uploadFiles[0].name : `${uploadFiles.length} files selected`}
                </Text>
              </Box>
            )}
          </VStack>

          <VStack
            align="stretch"
            spacing={4}
            w={{ base: "100%", xl: "320px" }}
            justify="space-between"
            p={4}
            borderRadius="8px"
            border="1px solid"
            borderColor={sidebarBorder}
            bg={sidebarBg}
          >
            <Box>
              <Text fontSize="sm" fontWeight="700" mb={2}>
                Destination project
              </Text>
              <VStack
                align="stretch"
                spacing={1}
                maxH="188px"
                minH="188px"
                overflowY="auto"
                p={2}
                borderRadius="8px"
                border="1px solid"
                borderColor={projectListBorder}
                bg={projectListBg}
                sx={{
                  "&::-webkit-scrollbar": { width: "8px" },
                  "&::-webkit-scrollbar-thumb": {
                    background: projectListBorder,
                    borderRadius: "999px",
                  },
                  "&::-webkit-scrollbar-track": {
                    background: "transparent",
                  },
                }}
              >
                {projects.map((project) => {
                  const isSelected = uploadProjectId === String(project.id);

                  return (
                    <Button
                      key={project.id}
                      type="button"
                      justifyContent="flex-start"
                      h="auto"
                      py={5}
                      px={3}
                      borderRadius="8px"
                      variant="ghost"
                      fontWeight={isSelected ? "700" : "600"}
                      whiteSpace="normal"
                      textAlign="left"
                      border="1px solid"
                      borderColor={isSelected ? selectedProjectBorder : "transparent"}
                      bg={isSelected ? selectedProjectBg : "transparent"}
                      boxShadow={isSelected ? selectedProjectShadow : "none"}
                      _hover={{ bg: isSelected ? selectedProjectBg : projectButtonHoverBg }}
                      _active={{ bg: isSelected ? selectedProjectBg : projectButtonHoverBg }}
                      onClick={() => onUploadProjectChange(String(project.id))}
                      isDisabled={projects.length === 0}
                    >
                      {project.name}
                    </Button>
                  );
                })}
              </VStack>
            </Box>

            {projects.length === 0 ? (
              <Text color={mutedText} fontSize="sm">
                Create a project first.
              </Text>
            ) : null}

            <Button
              type="submit"
              leftIcon={<BiUpload />}
              isDisabled={!uploadProjectId || uploadFiles.length === 0 || projects.length === 0}
              isLoading={submitting}
              mt={{ xl: "auto" }}
            >
              {submitting ? "Uploading..." : "Upload image"}
            </Button>
          </VStack>
        </Stack>
      </form>
    </GlassPanel>
  );
}
