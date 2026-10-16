import { useEffect, useMemo, useState } from "react";
import {
  Badge,
  Box,
  Button,
  HStack,
  Input,
  SimpleGrid,
  Stack,
  Text,
  VStack,
  useColorModeValue,
} from "@chakra-ui/react";
import { BiFolderOpen, BiTrash } from "react-icons/bi";

import { deletePipeline, listPipelines } from "../api/processes";
import { ConfirmDialog } from "../components/common/ConfirmDialog";
import { GlassPanel } from "../components/common/GlassPanel";
import { LoadingState, MessagePanel } from "../components/common/PageState";
import { PageHeader } from "../components/common/PageHeader";
import type { Pipeline } from "../types/api";
import { formatRelativeTime } from "../utils/date";
import { getErrorMessage } from "../utils/errors";
import { toImageUrl } from "../utils/images";

export default function PipelinesPage() {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deletingPipelineId, setDeletingPipelineId] = useState<string | null>(null);
  const [pendingDeletePipeline, setPendingDeletePipeline] = useState<Pipeline | null>(null);
  const [searchValue, setSearchValue] = useState("");

  const pageText = useColorModeValue("white", "white");
  const subtleText = useColorModeValue("rgba(245,241,235,0.72)", "whiteAlpha.700");
  const panelBg = useColorModeValue("transparent", "#151b23");
  const panelBorder = useColorModeValue("rgba(148,163,184,0.22)", "rgba(255,255,255,0.09)");
  const previewBg = useColorModeValue("#242424", "#1b2430");
  const compactPanelBg = useColorModeValue("#222222", "rgba(255,255,255,0.03)");
  const inputBg = useColorModeValue("#242424", "#1b2430");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");

      try {
        const data = await listPipelines();
        if (!cancelled) {
          setPipelines(data ?? []);
        }
      } catch (caughtError) {
        if (!cancelled) {
          setError(getErrorMessage(caughtError));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const sortedPipelines = useMemo(
    () => [...pipelines].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()),
    [pipelines],
  );
  const filteredPipelines = useMemo(() => {
    const query = searchValue.trim().toLowerCase();
    if (!query) {
      return sortedPipelines;
    }

    return sortedPipelines.filter((pipeline) => {
      const name = pipeline.name?.trim() || `Pipeline #${pipeline.id}`;
      return (
        name.toLowerCase().includes(query) ||
        pipeline.status.toLowerCase().includes(query) ||
        String(pipeline.id).includes(query)
      );
    });
  }, [searchValue, sortedPipelines]);

  function openPipeline(pipeline: Pipeline) {
    const params = new URLSearchParams({
      pipelineId: String(pipeline.id),
      projectId: String(pipeline.project_id),
      imageId: String(pipeline.source_image_id),
    });
    window.history.pushState({}, "", `/laboratory?${params.toString()}`);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  async function onDeletePipeline(pipelineId: string) {
    setDeletingPipelineId(pipelineId);
    setError("");
    try {
      await deletePipeline(pipelineId);
      setPipelines((prev) => prev.filter((pipeline) => pipeline.id !== pipelineId));
    } catch (caughtError) {
      setError(getErrorMessage(caughtError));
    } finally {
      setDeletingPipelineId(null);
    }
  }

  if (loading) {
    return <LoadingState label="Loading pipelines..." color={subtleText} />;
  }

  if (error) {
    return <MessagePanel message={error} tone="error" />;
  }

  return (
    <Stack spacing={6} color={pageText} px={{ base: 4, md: 6, xl: 8 }}>
      <ConfirmDialog
        isOpen={Boolean(pendingDeletePipeline)}
        title="Delete pipeline"
        description={`This will permanently remove ${
          pendingDeletePipeline?.name?.trim() || "this pipeline"
        }.`}
        confirmLabel="Delete pipeline"
        isLoading={deletingPipelineId === pendingDeletePipeline?.id}
        onClose={() => setPendingDeletePipeline(null)}
        onConfirm={() => {
          if (pendingDeletePipeline) {
            void onDeletePipeline(pendingDeletePipeline.id).then(() => setPendingDeletePipeline(null));
          }
        }}
      />

      <PageHeader
        eyebrow=""
        title="Pipelines"
        descriptionColor={subtleText}
        titleProps={{
          fontSize: { base: "3xl", md: "4xl" },
          textAlign: "left",
          fontWeight: "800",
          letterSpacing: "-0.05em",
        }}
      />

      <GlassPanel p={{ base: 4, md: 5 }} lightBg={panelBg} darkBg={panelBg} lightBorder={panelBorder} darkBorder={panelBorder}>
        <Stack direction={{ base: "column", lg: "row" }} align={{ base: "stretch", lg: "center" }} spacing={4}>
          <HStack spacing={3} flexWrap="wrap">
            <Badge width="fit-content" variant="subtle">
              {filteredPipelines.length} pipeline{filteredPipelines.length === 1 ? "" : "s"}
            </Badge>
            {sortedPipelines[0] ? (
              <Text color={subtleText} fontSize="sm">
                Latest update {formatRelativeTime(sortedPipelines[0].updated_at)}
              </Text>
            ) : null}
          </HStack>

          <Box flex="1" maxW={{ lg: "360px" }} ml={{ lg: "auto" }}>
            <Input
              value={searchValue}
              onChange={(event) => setSearchValue(event.target.value)}
              placeholder="Find a pipeline..."
              bg={inputBg}
              borderColor={panelBorder}
            />
          </Box>
        </Stack>
      </GlassPanel>

      {filteredPipelines.length === 0 ? (
        <GlassPanel p={5} lightBg={panelBg} darkBg={panelBg} lightBorder={panelBorder} darkBorder={panelBorder}>
          <Text color={subtleText}>{searchValue ? "No pipelines match your search." : "No pipelines available."}</Text>
        </GlassPanel>
      ) : (
        <SimpleGrid columns={{ base: 1, xl: 2 }} spacing={4}>
          {filteredPipelines.map((pipeline) => (
            <GlassPanel
              key={pipeline.id}
              p={{ base: 4, md: 4 }}
              lightBg={compactPanelBg}
              darkBg={compactPanelBg}
              lightBorder={panelBorder}
              darkBorder={panelBorder}
            >
              <VStack align="stretch" spacing={4}>
                <HStack justify="space-between" align="start" gap={3}>
                  <VStack align="start" spacing={1} minW={0}>
                    <Text fontWeight="800" fontSize="lg" letterSpacing="-0.04em" noOfLines={1}>
                      {pipeline.name?.trim() || `Pipeline #${pipeline.id}`}
                    </Text>
                    <HStack spacing={2} flexWrap="wrap">
                      <Badge variant="subtle">{pipeline.status}</Badge>
                      <Text color={subtleText} fontSize="sm">
                        {formatRelativeTime(pipeline.updated_at)}
                      </Text>
                    </HStack>
                  </VStack>
                  <HStack spacing={2} flexShrink={0}>
                    <Button size="sm" leftIcon={<BiFolderOpen />} onClick={() => openPipeline(pipeline)}>
                      Open
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      leftIcon={<BiTrash />}
                      onClick={() => setPendingDeletePipeline(pipeline)}
                      isLoading={deletingPipelineId === pipeline.id}
                    >
                      Delete
                    </Button>
                  </HStack>
                </HStack>

                <SimpleGrid columns={2} spacing={3}>
                  <ImagePanel
                    label="Start"
                    src={toImageUrl(pipeline.start_image_url)}
                    alt="Pipeline start"
                    panelBorder={panelBorder}
                    previewBg={previewBg}
                  />
                  <ImagePanel
                    label="Final"
                    src={pipeline.final_image_url ? toImageUrl(pipeline.final_image_url) : ""}
                    alt="Pipeline final"
                    panelBorder={panelBorder}
                    previewBg={previewBg}
                    emptyLabel="Not completed yet"
                  />
                </SimpleGrid>
              </VStack>
            </GlassPanel>
          ))}
        </SimpleGrid>
      )}
    </Stack>
  );
}

function ImagePanel({
  label,
  src,
  alt,
  panelBorder,
  previewBg,
  emptyLabel,
}: {
  label: string;
  src: string;
  alt: string;
  panelBorder: string;
  previewBg: string;
  emptyLabel?: string;
}) {
  const mutedText = useColorModeValue("rgba(245,241,235,0.72)", "whiteAlpha.700");

  return (
    <Box minW={0}>
      <Text
        color={mutedText}
        fontSize="xs"
        textTransform="uppercase"
        mb={2}
        letterSpacing="0.14em"
      >
        {label}
      </Text>
      <Box
        borderRadius="8px"
        overflow="hidden"
        border="1px solid"
        borderColor={panelBorder}
        h={{ base: "120px", md: "136px" }}
        display="flex"
        alignItems="center"
        justifyContent="center"
        bg={previewBg}
      >
        {src ? (
          <img
            src={src}
            alt={alt}
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : (
          <Text color={mutedText} fontSize="sm">
            {emptyLabel}
          </Text>
        )}
      </Box>
    </Box>
  );
}
