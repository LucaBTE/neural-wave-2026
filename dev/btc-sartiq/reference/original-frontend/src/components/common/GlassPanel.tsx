import { Box, type BoxProps, useColorModeValue } from "@chakra-ui/react";

interface GlassPanelProps extends BoxProps {
  lightBg?: string;
  darkBg?: string;
  lightBorder?: string;
  darkBorder?: string;
  shadow?: string;
  darkShadow?: string;
}

export function GlassPanel({
  children,
  lightBg = "#202020",
  darkBg = "#151b23",
  lightBorder = "rgba(255, 255, 255, 0.1)",
  darkBorder = "rgba(240, 246, 252, 0.1)",
  shadow = "none",
  darkShadow = "none",
  ...boxProps
}: GlassPanelProps) {
  const bg = useColorModeValue(lightBg, darkBg);
  const borderColor = useColorModeValue(lightBorder, darkBorder);
  const boxShadow = useColorModeValue(shadow, darkShadow);

  return (
    <Box
      border="1px solid"
      borderColor={borderColor}
      bg={bg}
      backdropFilter="blur(18px)"
      boxShadow={boxShadow}
      borderRadius="8px"
      {...boxProps}
    >
      {children}
    </Box>
  );
}
