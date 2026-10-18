import { Box, Button } from "@chakra-ui/react";

import type { OpenedImage } from "../../types/ui";

interface ImageLightboxProps {
  openedImage: OpenedImage | null;
  onClose: () => void;
}

export function ImageLightbox({ openedImage, onClose }: ImageLightboxProps) {
  if (!openedImage) {
    return null;
  }

  return (
    <Box
      position="fixed"
      inset={0}
      zIndex={2000}
      bg="blackAlpha.900"
      display="flex"
      alignItems="center"
      justifyContent="center"
      p={{ base: 3, md: 6 }}
      onClick={onClose}
    >
      <Button
        type="button"
        position="absolute"
        top={4}
        right={4}
        size="sm"
        colorScheme="whiteAlpha"
        onClick={onClose}
      >
        ×
      </Button>
      <Box
        maxW="95vw"
        maxH="92vh"
        borderRadius="8px"
        overflow="hidden"
        border="1px solid"
        borderColor="whiteAlpha.400"
        bg="black"
        onClick={(event) => event.stopPropagation()}
      >
        <img
          src={openedImage.src}
          alt={openedImage.name}
          style={{ maxWidth: "95vw", maxHeight: "92vh", width: "auto", height: "auto", display: "block" }}
        />
      </Box>
    </Box>
  );
}
