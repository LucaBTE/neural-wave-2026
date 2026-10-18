import {
  Badge,
  Box,
  Button,
  ButtonGroup,
  FormControl,
  FormLabel,
  HStack,
  Input,
  Select,
  Spinner,
  Stack,
  Switch,
  Text,
  VStack,
  useColorModeValue,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { BiChevronRight, BiRefresh, BiSave, BiTrash } from "react-icons/bi";

import type { ConvexHullPreviewMode, LabCell } from "../../types/laboratory";
import { getSelectedModelOption, getVisibleAdditionalSettings } from "../../hooks/useLaboratoryNotebook";
import { toImageUrl } from "../../utils/images";

interface LabCellCardProps {
  cell: LabCell;
  index: number;
  cellsLength: number;
  inputUrl: string;
  maskOverlayUrl?: string;
  panelBorder: string;
  outputBg: string;
  subtleText: string;
  runningAll: boolean;
  savingCellId: string;
  saveMessage?: string;
  saveError?: string;
  runCellLabel: string;
  resetFromLabel: string;
  removeCellLabel: string;
  waitingOutputLabel: string;
  saveToProjectLabel: string;
  onRunCell: () => void;
  onReset: () => void;
  onRemove: () => void;
  onSetSegmentOutputConvexHull: (enabled: boolean) => void;
  onSetSegmentOutputConvexHullMode: (mode: ConvexHullPreviewMode) => void;
  onUpdateCell: (patch: Partial<LabCell>) => void;
  onUpdateModel: (modelKey: string) => void;
  onUpdateAdditionalSetting: (settingKey: string, value: string | number | boolean) => void;
  onSaveOutput: () => void;
}

export function LabCellCard({
  cell,
  index,
  cellsLength,
  inputUrl,
  maskOverlayUrl,
  panelBorder,
  outputBg,
  subtleText,
  runningAll,
  savingCellId,
  saveMessage,
  saveError,
  runCellLabel,
  resetFromLabel,
  removeCellLabel,
  waitingOutputLabel,
  saveToProjectLabel,
  onRunCell,
  onReset,
  onRemove,
  onSetSegmentOutputConvexHull,
  onSetSegmentOutputConvexHullMode,
  onUpdateCell,
  onUpdateModel,
  onUpdateAdditionalSetting,
  onSaveOutput,
}: LabCellCardProps) {
  const cardBg = useColorModeValue("#222222", "rgba(255,255,255,0.03)");
  const selectedModel = getSelectedModelOption(cell);
  const segmentOutputMode = !cell.outputConvexHullEnabled
    ? "original"
    : cell.outputConvexHullMode === "rectangle"
      ? "rectangle"
      : "simple";
  const settingDefinitions = getVisibleAdditionalSettings(
    selectedModel?.additional_settings ?? [],
    cell.additionalSettings,
  ).filter((setting) => !["use_convex_hull_mask", "convex_hull_expand_px"].includes(setting.key));
  const [showMaskOverlay, setShowMaskOverlay] = useState(true);
  const isWorking = cell.status === "running" || cell.outputPreviewLoading;
  const supportsMaskOverlay =
    cell.processType === "generate_from_prompt" || cell.processType === "remove_with_mask";

  useEffect(() => {
    if (!supportsMaskOverlay || !maskOverlayUrl) {
      setShowMaskOverlay(false);
      return;
    }

    setShowMaskOverlay(true);
  }, [maskOverlayUrl, supportsMaskOverlay]);

  return (
    <Box
      p={4}
      borderRadius="8px"
      border="1px solid"
      borderColor={panelBorder}
      bg={cardBg}
    >
      <HStack justify="space-between" align="center" flexWrap="wrap" mb={3}>
        <HStack>
          <Badge>{`Cell ${index + 1}`}</Badge>
          <Text fontWeight="800" letterSpacing="-0.03em">
            {cell.title}
          </Text>
          <Badge variant="subtle">{cell.status}</Badge>
        </HStack>

        <HStack>
          <Button
            size="sm"
            leftIcon={<BiChevronRight />}
            onClick={onRunCell}
            isLoading={cell.status === "running"}
            isDisabled={runningAll}
          >
            {runCellLabel}
          </Button>
          <Button size="sm" variant="outline" leftIcon={<BiRefresh />} onClick={onReset} isDisabled={index === cellsLength - 1}>
            {resetFromLabel}
          </Button>
          <Button size="sm" variant="outline" leftIcon={<BiTrash />} onClick={onRemove} isDisabled={index === 0}>
            {removeCellLabel}
          </Button>
        </HStack>
      </HStack>

      <Stack direction={{ base: "column", xl: "row" }} spacing={4}>
        <VStack align="stretch" flex={1} spacing={3}>
          <Box
            position="relative"
            borderRadius="6px"
            overflow="hidden"
            border="1px solid"
            borderColor={panelBorder}
            bg={outputBg}
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            {inputUrl ? (
              <img
                src={inputUrl}
                alt={`Cell ${index + 1} input`}
                style={{ width: "100%", height: "auto", display: "block" }}
              />
            ) : null}
            {inputUrl && maskOverlayUrl && supportsMaskOverlay && showMaskOverlay ? (
              <Box
                as="img"
                src={maskOverlayUrl}
                alt={`Cell ${index + 1} mask overlay`}
                position="absolute"
                inset={0}
                w="100%"
                h="100%"
                objectFit="contain"
                opacity={0.92}
                pointerEvents="none"
                mixBlendMode="screen"
              />
            ) : null}
          </Box>
          {maskOverlayUrl && supportsMaskOverlay ? (
            <VStack align="stretch" spacing={3}>
              <FormControl display="flex" alignItems="center" justifyContent="space-between">
                <Box pr={4}>
                  <FormLabel mb={0} fontSize="sm">
                    Show mask overlay
                  </FormLabel>
                  <Text color={subtleText} fontSize="xs">
                    Overlay the white masked area on top of the process input preview.
                  </Text>
                </Box>
                <Switch
                  isChecked={showMaskOverlay}
                  onChange={(event) => setShowMaskOverlay(event.target.checked)}
                  isDisabled={cell.status === "running" || runningAll}
                />
              </FormControl>
            </VStack>
          ) : null}
          {cell.promptRequired ? (
            <Input
              placeholder="Write prompt..."
              value={cell.prompt}
              onChange={(event) => onUpdateCell({ prompt: event.target.value })}
              isDisabled={cell.status === "running" || runningAll}
            />
          ) : null}

          {cell.modelOptions.length > 0 ? (
            <Select value={cell.modelKey} onChange={(event) => onUpdateModel(event.target.value)} isDisabled={cell.status === "running" || runningAll}>
              {cell.modelOptions.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </Select>
          ) : null}

          {settingDefinitions.length > 0 ? (
            <VStack align="stretch" spacing={3}>
              <Text fontSize="sm" fontWeight="semibold">
                Additional settings
              </Text>

              {settingDefinitions.map((setting) => {
                const currentValue = cell.additionalSettings[setting.key] ?? setting.default_value;

                if (setting.type === "boolean") {
                  return (
                    <FormControl key={setting.key} display="flex" alignItems="center" justifyContent="space-between">
                      <Box pr={4}>
                        <FormLabel mb={0} fontSize="sm">
                          {setting.label}
                        </FormLabel>
                        {setting.description ? (
                          <Text color={subtleText} fontSize="xs">
                            {setting.description}
                          </Text>
                        ) : null}
                      </Box>
                      <Switch
                        isChecked={Boolean(currentValue)}
                        onChange={(event) => onUpdateAdditionalSetting(setting.key, event.target.checked)}
                        isDisabled={cell.status === "running" || runningAll}
                      />
                    </FormControl>
                  );
                }

                if (setting.type === "select") {
                  const isRemovalModeButtons =
                    setting.key === "mode" &&
                    cell.processType === "remove_with_mask" &&
                    (setting.options ?? []).length === 3;

                  if (isRemovalModeButtons) {
                    return (
                      <FormControl key={setting.key}>
                        <FormLabel fontSize="sm">{setting.label}</FormLabel>
                        <ButtonGroup isAttached size="sm" variant="outline">
                          {(setting.options ?? []).map((option) => (
                            <Button
                              key={option.value}
                              variant={currentValue === option.value ? "solid" : "outline"}
                              onClick={() => onUpdateAdditionalSetting(setting.key, option.value)}
                              isDisabled={cell.status === "running" || runningAll}
                            >
                              {option.label}
                            </Button>
                          ))}
                        </ButtonGroup>
                        {setting.description ? (
                          <Text color={subtleText} fontSize="xs" mt={1}>
                            {setting.description}
                          </Text>
                        ) : null}
                      </FormControl>
                    );
                  }

                  return (
                    <FormControl key={setting.key}>
                      <FormLabel fontSize="sm">{setting.label}</FormLabel>
                      <Select
                        value={String(currentValue ?? "")}
                        onChange={(event) => onUpdateAdditionalSetting(setting.key, event.target.value)}
                        isDisabled={cell.status === "running" || runningAll}
                      >
                        {(setting.options ?? []).map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </Select>
                      {setting.description ? (
                        <Text color={subtleText} fontSize="xs" mt={1}>
                          {setting.description}
                        </Text>
                      ) : null}
                    </FormControl>
                  );
                }

                if (setting.type === "text") {
                  return (
                    <FormControl key={setting.key}>
                      <FormLabel fontSize="sm">{setting.label}</FormLabel>
                      <Input
                        value={String(currentValue ?? "")}
                        onChange={(event) => onUpdateAdditionalSetting(setting.key, event.target.value)}
                        isDisabled={cell.status === "running" || runningAll}
                      />
                      {setting.description ? (
                        <Text color={subtleText} fontSize="xs" mt={1}>
                          {setting.description}
                        </Text>
                      ) : null}
                    </FormControl>
                  );
                }

                return (
                  <FormControl key={setting.key}>
                    <FormLabel fontSize="sm">{setting.label}</FormLabel>
                    <Input
                      type="number"
                      min={setting.min_value}
                      max={setting.max_value}
                      step={setting.step ?? 1}
                      value={String(currentValue ?? "")}
                      onChange={(event) =>
                        onUpdateAdditionalSetting(
                          setting.key,
                          event.target.value === "" ? "" : Number(event.target.value),
                        )
                      }
                      isDisabled={cell.status === "running" || runningAll}
                    />
                    {setting.description ? (
                      <Text color={subtleText} fontSize="xs" mt={1}>
                        {setting.description}
                      </Text>
                    ) : null}
                  </FormControl>
                );
              })}
            </VStack>
          ) : null}
        </VStack>

        <VStack align="stretch" flex={1} spacing={2}>
          <Box
            role="group"
            position="relative"
            borderRadius="6px"
            overflow="hidden"
            border="1px solid"
            borderColor={panelBorder}
            bg={outputBg}
            display="flex"
            alignItems="center"
            justifyContent="center"
            minH={cell.outputUrl ? undefined : { base: "220px", md: "280px" }}
          >
            {cell.outputUrl ? (
              <>
                <Box
                  as="img"
                  src={toImageUrl(cell.outputUrl)}
                  alt={`Cell ${index + 1} output`}
                  w="100%"
                  h="auto"
                  display="block"
                  transition="filter 0.18s ease, transform 0.18s ease"
                  _groupHover={{
                    filter: "blur(0px)",
                    transform: "scale(1.01)",
                  }}
                />
                {isWorking ? (
                  <VStack
                    position="absolute"
                    inset={0}
                    bg="blackAlpha.700"
                    justify="center"
                    align="center"
                    spacing={3}
                    zIndex={1}
                  >
                    <Spinner size="xl" thickness="4px" color="white" />
                    <Text color="white" fontSize="sm" fontWeight="medium">
                      {cell.outputPreviewLoading ? "Building convex hull preview..." : "Processing image..."}
                    </Text>
                  </VStack>
                ) : null}
                <Box
                  position="absolute"
                  top={3}
                  right={3}
                  opacity={{ base: 1, md: 0 }}
                  transition="opacity 0.18s ease"
                  _groupHover={{ opacity: 1 }}
                >
                  <Button
                    size="sm"
                    colorScheme="blackAlpha"
                    bg="blackAlpha.700"
                    color="white"
                    _hover={{ bg: "blackAlpha.800" }}
                    _active={{ bg: "blackAlpha.900" }}
                    backdropFilter="blur(8px)"
                    borderRadius="6px"
                    onClick={onSaveOutput}
                    isLoading={savingCellId === cell.id}
                    leftIcon={<BiSave />}
                  >
                    {saveToProjectLabel}
                  </Button>
                </Box>
              </>
            ) : (
              <VStack spacing={3}>
                {isWorking ? (
                  <>
                    <Spinner size="xl" thickness="4px" />
                    <Text color={subtleText} fontSize="sm">
                      {cell.outputPreviewLoading ? "Building convex hull preview..." : "Processing image..."}
                    </Text>
                  </>
                ) : (
                  <Text color={subtleText} fontSize="sm">
                    {waitingOutputLabel}
                  </Text>
                )}
              </VStack>
            )}
          </Box>

          {saveMessage ? (
            <Text color="green.400" fontSize="sm">
              {saveMessage}
            </Text>
          ) : null}

          {saveError ? (
            <Text color="red.400" fontSize="sm">
              {saveError}
            </Text>
          ) : null}

          {cell.processType === "segment_from_prompt" && cell.originalOutputUrl ? (
            <VStack align="stretch" spacing={3} pt={2}>
              <FormControl>
                <Box pr={4} mb={2}>
                  <FormLabel mb={0} fontSize="sm">
                    Output mask mode
                  </FormLabel>
                  <Text color={subtleText} fontSize="xs">
                    Choose the original mask, the light convex hull, or the bounding box.
                  </Text>
                </Box>
                <ButtonGroup isAttached size="sm" variant="outline">
                  <Button
                    variant={segmentOutputMode === "original" ? "solid" : "outline"}
                    onClick={() => onSetSegmentOutputConvexHull(false)}
                    isDisabled={isWorking || runningAll}
                  >
                    Original
                  </Button>
                  <Button
                    variant={segmentOutputMode === "simple" ? "solid" : "outline"}
                    onClick={() => onSetSegmentOutputConvexHullMode("simple")}
                    isDisabled={isWorking || runningAll}
                  >
                    ConvexHull
                  </Button>
                  <Button
                    variant={segmentOutputMode === "rectangle" ? "solid" : "outline"}
                    onClick={() => onSetSegmentOutputConvexHullMode("rectangle")}
                    isDisabled={isWorking || runningAll}
                  >
                    Box
                  </Button>
                </ButtonGroup>
              </FormControl>
            </VStack>
          ) : null}

          {cell.error ? (
            <Text color="red.400" fontSize="sm">
              {cell.error}
            </Text>
          ) : null}
        </VStack>
      </Stack>
    </Box>
  );
}
