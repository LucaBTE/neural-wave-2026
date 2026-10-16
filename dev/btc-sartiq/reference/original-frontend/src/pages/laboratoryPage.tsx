import {
  Badge,
  Box,
  Button,
  HStack,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Spinner,
  Stack,
  Text,
  VStack,
  useColorModeValue,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { BiPlayCircle, BiSave } from "react-icons/bi";

import { AddProcessControl } from "../components/laboratory/AddProcessControl";
import { LabCellCard } from "../components/laboratory/LabCellCard";
import { GlassPanel } from "../components/common/GlassPanel";
import { PageHeader } from "../components/common/PageHeader";
import { StatusNotice } from "../components/common/StatusNotice";
import { useLaboratoryNotebook } from "../hooks/useLaboratoryNotebook";

type ExplanationGroup = {
  text: string;
  subitems: string[];
};

function parseNotebookExplanationList(items: string[]) {
  const intro: string[] = [];
  const bullets: ExplanationGroup[] = [];
  const outro: string[] = [];

  let currentGroup: ExplanationGroup | null = null;
  let bulletSectionStarted = false;

  for (const item of items) {
    const clean = item.trim();
    if (!clean) continue;

    if (clean.startsWith("- - ")) {
      if (currentGroup) {
        currentGroup.subitems.push(clean.slice(4).trim());
      }
      continue;
    }

    if (clean.startsWith("- ")) {
      currentGroup = {
        text: clean.slice(2).trim(),
        subitems: [],
      };
      bullets.push(currentGroup);
      bulletSectionStarted = true;
      continue;
    }

    if (bulletSectionStarted) {
      outro.push(clean);
    } else {
      intro.push(clean);
    }
  }

  return { intro, bullets, outro };
}

export default function LaboratoryPage() {
  const pageText = useColorModeValue("white", "white");
  const subtleText = useColorModeValue("rgba(245,241,235,0.72)", "whiteAlpha.700");
  const panelBg = useColorModeValue("transparent", "#151b23");
  const panelBorder = useColorModeValue("rgba(148,163,184,0.2)", "rgba(255,255,255,0.09)");
  const outputBg = useColorModeValue("#242424", "#1b2430");
  const modalBg = useColorModeValue("#222222", "#151b23");
  const modalBorder = useColorModeValue("rgba(255,255,255,0.12)", "rgba(240,246,252,0.12)");
  const modalOverlay = useColorModeValue("rgba(226,232,240,0.62)", "rgba(4, 6, 12, 0.74)");
  const modalMuted = useColorModeValue("rgba(245,241,235,0.7)", "rgba(226,232,240,0.7)");
  const closeBg = useColorModeValue("#242424", "#1b2430");
  const closeBorder = useColorModeValue("rgba(255,255,255,0.12)", "rgba(240,246,252,0.12)");
  const overwritePanelBg = useColorModeValue("#242424", "#1b2430");

  const {
    queryProjectId,
    queryImageId,
    selectedImage,
    activePipelineId,
    cells,
    runningAll,
    savingPipeline,
    savingCellId,
    saveMessageByCell,
    saveErrorByCell,
    saveMessage,
    saveError,
    loadingPipeline,
    currentPipelineName,
    notebookExplanationList,
    getAvailableProcessesAfter,
    getSelectedProcessTypeFor,
    canAddProcessAfter,
    setSelectedProcessTypeFor,
    addCell,
    updateCell,
    updateModelForCell,
    updateAdditionalSetting,
    getInputForCell,
    getMaskOverlayUrlForCell,
    resetFromCell,
    removeCell,
    runCell,
    runAllCells,
    savePipeline,
    saveCellOutputToProject,
    setSegmentOutputConvexHull,
    setSegmentOutputConvexHullMode,
  } = useLaboratoryNotebook();

  const notebookExplanation = parseNotebookExplanationList(notebookExplanationList);
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
  const [pipelineName, setPipelineName] = useState(currentPipelineName);

  useEffect(() => {
    if (!isSaveModalOpen) {
      setPipelineName(currentPipelineName);
    }
  }, [currentPipelineName, isSaveModalOpen]);

  async function onSavePipeline(mode: "overwrite" | "save_as_new") {
    const saved = await savePipeline(mode, pipelineName);
    if (saved) {
      setIsSaveModalOpen(false);
    }
  }

  return (
    <Stack spacing={6} color={pageText} px={{ base: 4, md: 6, xl: 8 }}>
      <PageHeader
        eyebrow=""
        title="Laboratory"
        description="Notebook workflow."
        descriptionColor={subtleText}
        titleProps={{
          fontSize: { base: "3xl", md: "4xl" },
          textAlign: "left",
          fontWeight: "800",
          letterSpacing: "-0.05em",
        }}
      >
        {notebookExplanationList.length > 0 ? (
          <VStack align="start" spacing={1} mt={2}>
            {notebookExplanation.intro.map((itemText) => (
              <Text key={itemText} color={subtleText} fontSize="sm">
                {itemText}
              </Text>
            ))}

            {notebookExplanation.bullets.length > 0 ? (
              <VStack as="ul" align="start" spacing={1} pl={5} mt={1}>
                {notebookExplanation.bullets.map((item) => (
                  <Box as="li" key={item.text} color={subtleText} fontSize="sm">
                    <Text as="span">{item.text}</Text>
                    {item.subitems.length > 0 ? (
                      <VStack as="ul" align="start" spacing={1} pl={5} mt={1}>
                        {item.subitems.map((subitem) => (
                          <Box as="li" key={subitem} color={subtleText} fontSize="sm">
                            <Text as="span">{subitem}</Text>
                          </Box>
                        ))}
                      </VStack>
                    ) : null}
                  </Box>
                ))}
              </VStack>
            ) : null}

            {notebookExplanation.outro.map((itemText) => (
              <Text key={itemText} color={subtleText} fontSize="sm" pt={1}>
                {itemText}
              </Text>
            ))}
          </VStack>
        ) : null}
      </PageHeader>

      <Stack direction={{ base: "column", lg: "row" }} justify="space-between" align={{ base: "stretch", lg: "center" }} spacing={4}>
          <VStack align="start" spacing={2}>
          </VStack>
          <HStack spacing={3} flexWrap="wrap">
            <Button variant="outline" leftIcon={<BiSave />} onClick={() => setIsSaveModalOpen(true)} isDisabled={runningAll}>
              Save pipeline
            </Button>
            <Button leftIcon={<BiPlayCircle />} onClick={() => void runAllCells()} isLoading={runningAll}>
              Run all cells
            </Button>
          </HStack>
        </Stack>

      {saveMessage ? <StatusNotice tone="success" variant="text">{saveMessage}</StatusNotice> : null}
      {saveError ? <StatusNotice tone="error" variant="text">{saveError}</StatusNotice> : null}

      <GlassPanel p={5} lightBg={panelBg} darkBg={panelBg} lightBorder={panelBorder} darkBorder={panelBorder}>
        <VStack align="stretch" spacing={4}>
          <HStack justify="space-between" align="center" flexWrap="wrap" gap={3}>
            <VStack align="start" spacing={1}>
              <Text fontWeight="800" fontSize={{ base: "lg", md: "xl" }} letterSpacing="-0.04em">
                Workflow cells
              </Text>
              <Text color={subtleText} fontSize="sm">
                {selectedImage?.fileName ?? "No input image selected"}
              </Text>
            </VStack>
            <Badge width="fit-content" variant="subtle">
              {cells.length} cell{cells.length === 1 ? "" : "s"}
            </Badge>
          </HStack>

          {loadingPipeline ? (
            <HStack color={subtleText}>
              <Spinner size="sm" />
              <Text fontSize="sm">Loading saved pipeline...</Text>
            </HStack>
          ) : null}

          {!selectedImage ? (
            <Text color={subtleText}>No image selected yet. Open this page from Home &gt; Edit.</Text>
          ) : (
            <VStack align="stretch" spacing={5} mt={2}>
              {cells.map((cell, index) => (
                <VStack key={cell.id} align="stretch" spacing={4}>
                  <LabCellCard
                    cell={cell}
                    index={index}
                    cellsLength={cells.length}
                    inputUrl={getInputForCell(index, cells)}
                    maskOverlayUrl={getMaskOverlayUrlForCell(index, cells) || undefined}
                    panelBorder={panelBorder}
                    outputBg={outputBg}
                    subtleText={subtleText}
                    runningAll={runningAll}
                    savingCellId={savingCellId}
                    saveMessage={saveMessageByCell[cell.id]}
                    saveError={saveErrorByCell[cell.id]}
                    runCellLabel="Run cell"
                    resetFromLabel="Reset from here"
                    removeCellLabel="Remove cell"
                    waitingOutputLabel="Output will appear here"
                    saveToProjectLabel="Save to project"
                    onRunCell={() => void runCell(index)}
                    onReset={() => resetFromCell(index)}
                    onRemove={() => removeCell(index)}
                    onSetSegmentOutputConvexHull={(enabled) => void setSegmentOutputConvexHull(index, enabled)}
                    onSetSegmentOutputConvexHullMode={(mode) =>
                      void setSegmentOutputConvexHullMode(index, mode)
                    }
                    onUpdateCell={(patch) => updateCell(index, patch)}
                    onUpdateModel={(modelKey) => updateModelForCell(index, modelKey)}
                    onUpdateAdditionalSetting={(settingKey, value) =>
                      updateAdditionalSetting(index, settingKey, value)
                    }
                    onSaveOutput={() => void saveCellOutputToProject(cell)}
                  />

                  {index === cells.length - 1 ? (
                    <AddProcessControl
                      availableProcesses={getAvailableProcessesAfter(index)}
                      selectedProcessType={getSelectedProcessTypeFor(index)}
                      isAddDisabled={!canAddProcessAfter(index)}
                      onProcessTypeChange={(value) => setSelectedProcessTypeFor(index, value)}
                      onAdd={() => addCell(index)}
                    />
                  ) : null}
                </VStack>
              ))}

              {cells.length === 0 ? (
                <VStack spacing={3} py={4}>
                  <Text color={subtleText} fontSize="sm">
                    Add your first work cell to start your job
                  </Text>
                  <AddProcessControl
                    availableProcesses={getAvailableProcessesAfter(-1)}
                    selectedProcessType={getSelectedProcessTypeFor(-1)}
                    isAddDisabled={!canAddProcessAfter(-1)}
                    onProcessTypeChange={(value) => setSelectedProcessTypeFor(-1, value)}
                    onAdd={() => addCell(-1)}
                  />
                </VStack>
              ) : null}
            </VStack>
          )}
        </VStack>
      </GlassPanel>

      <Modal isOpen={isSaveModalOpen} onClose={() => setIsSaveModalOpen(false)} isCentered>
        <ModalOverlay bg={modalOverlay} backdropFilter="blur(10px)" />
        <ModalContent
          bg={modalBg}
          color={pageText}
          border="1px solid"
          borderColor={modalBorder}
          borderRadius="8px"
          boxShadow="none"
          backdropFilter="blur(8px)"
          px={{ base: 1, md: 2 }}
        >
          <ModalHeader pt={7} pb={2} fontSize={{ base: "2xl", md: "3xl" }} fontWeight="700" textAlign="center">
            {activePipelineId ? "Save pipeline changes" : "Save pipeline"}
          </ModalHeader>
          <ModalCloseButton
            top={4}
            right={4}
            borderRadius="6px"
            bg={closeBg}
            border="1px solid"
            borderColor={closeBorder}
            _hover={{ borderColor: modalBorder }}
          />
          <ModalBody pb={3}>
            <VStack align="stretch" spacing={4}>
              <Text color={modalMuted} fontSize="md" textAlign="center" maxW="420px" mx="auto">
                {activePipelineId
                  ? "Overwrite the current pipeline or create a new one from the current notebook."
                  : "Choose a name for this pipeline."}
              </Text>
              <Input
                placeholder="Pipeline name"
                value={pipelineName}
                onChange={(event) => setPipelineName(event.target.value)}
                size="lg"
              />
              {activePipelineId ? (
                <Box
                  border="1px solid"
                  borderColor={modalBorder}
                  bg={overwritePanelBg}
                  borderRadius="8px"
                  p={4}
                >
                  <Text fontWeight="700" mb={1}>
                    Overwrite existing pipeline
                  </Text>
                  <Text color={modalMuted} fontSize="sm">
                    Use overwrite to replace the saved steps and current output of this pipeline.
                  </Text>
                </Box>
              ) : null}
            </VStack>
          </ModalBody>
          <ModalFooter pt={2} pb={7}>
            <HStack
              w="full"
              justify={{ base: "stretch", md: "space-between" }}
              align="center"
              spacing={3}
              flexDirection={{ base: "column-reverse", md: "row" }}
            >
              <Button
                variant="ghost"
                onClick={() => setIsSaveModalOpen(false)}
                isDisabled={savingPipeline}
                w={{ base: "full", md: "auto" }}
              >
                Cancel
              </Button>
              <HStack w={{ base: "full", md: "auto" }} spacing={3}>
                {activePipelineId ? (
                  <Button
                    variant="outline"
                    onClick={() => void onSavePipeline("save_as_new")}
                    isLoading={savingPipeline}
                    w={{ base: "full", md: "auto" }}
                  >
                    Save as new
                  </Button>
                ) : null}
                <Button
                  onClick={() => void onSavePipeline(activePipelineId ? "overwrite" : "save_as_new")}
                  isLoading={savingPipeline}
                  w={{ base: "full", md: "auto" }}
                >
                  {activePipelineId ? "Overwrite" : "Save pipeline"}
                </Button>
              </HStack>
            </HStack>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Stack>
  );
}
