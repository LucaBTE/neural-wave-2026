import { extendTheme, type ThemeConfig } from "@chakra-ui/react";

const config: ThemeConfig = {
  initialColorMode: "light",
  useSystemColorMode: false,
};

const theme = extendTheme({
  config,
  fonts: {
    heading: "'Manrope', 'Segoe UI', sans-serif",
    body: "'Manrope', 'Segoe UI', sans-serif",
  },
  styles: {
    global: {
      body: {
        bg: "transparent",
      },
    },
  },
  components: {
    Button: {
      baseStyle: {
        borderRadius: "6px",
        fontWeight: "700",
        letterSpacing: "-0.01em",
        borderWidth: "1px",
      },
      sizes: {
        sm: {
          h: "36px",
          px: 3.5,
          fontSize: "sm",
        },
        md: {
          h: "42px",
          px: 4,
          fontSize: "sm",
        },
      },
      variants: {
        solid: {
          bg: "#2e2e2e",
          color: "#f5f1eb",
          borderColor: "rgba(255, 255, 255, 0.16)",
          boxShadow: "none",
          _hover: { bg: "#343434", borderColor: "rgba(255, 255, 255, 0.24)" },
          _active: { bg: "#383838" },
          _dark: {
            bg: "#1b2430",
            color: "#f0f6fc",
            borderColor: "rgba(240, 246, 252, 0.16)",
            boxShadow: "none",
            _hover: {
              bg: "#24303d",
              borderColor: "rgba(240, 246, 252, 0.24)",
            },
            _active: { bg: "#16202a" },
          },
        },
        outline: {
          borderWidth: "1px",
          borderColor: "rgba(255, 255, 255, 0.14)",
          bg: "#232323",
          color: "inherit",
          boxShadow: "none",
          _hover: { bg: "#292929", borderColor: "rgba(255, 255, 255, 0.22)" },
          _active: { bg: "#2d2d2d" },
          _dark: {
            borderColor: "rgba(240, 246, 252, 0.16)",
            bg: "#161b22",
            _hover: {
              bg: "#1c2128",
              borderColor: "rgba(240, 246, 252, 0.24)",
            },
            _active: { bg: "rgba(255,255,255,0.07)" },
          },
        },
        ghost: {
          color: "inherit",
          boxShadow: "none",
          _hover: { bg: "rgba(255,255,255,0.04)" },
          _active: { bg: "rgba(255,255,255,0.06)" },
        },
      },
      defaultProps: {
        variant: "solid",
      },
    },
    Input: {
      defaultProps: {
        variant: "filled",
      },
      variants: {
        filled: {
          field: {
            borderRadius: "6px",
            bg: "rgba(255,255,255,0.02)",
            color: "#f5f1eb",
            borderWidth: "1px",
            borderStyle: "solid",
            borderColor: "rgba(255,255,255,0.08)",
            boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.08)",
            _placeholder: {
              color: "rgba(245,241,235,0.36)",
            },
            _hover: {
              bg: "rgba(255,255,255,0.04)",
              borderColor: "rgba(255,255,255,0.12)",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.12)",
            },
            _focusVisible: {
              bg: "rgba(255,255,255,0.04)",
              borderColor: "rgba(255,255,255,0.12)",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.12)",
            },
            _dark: {
              bg: "#151b23",
              border: "1px solid rgba(240,246,252,0.14)",
              _hover: { bg: "#1b2430", borderColor: "rgba(240,246,252,0.22)" },
              _focusVisible: {
                bg: "#1b2430",
                borderColor: "rgba(240,246,252,0.3)",
                boxShadow: "0 0 0 1px rgba(240,246,252,0.18)",
              },
            },
          },
        },
      },
    },
    Select: {
      variants: {
        filled: {
          field: {
            borderRadius: "6px",
            bg: "rgba(255,255,255,0.02)",
            color: "#f5f1eb",
            borderWidth: "1px",
            borderStyle: "solid",
            borderColor: "rgba(255,255,255,0.08)",
            boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.08)",
            _hover: {
              bg: "rgba(255,255,255,0.04)",
              borderColor: "rgba(255,255,255,0.12)",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.12)",
            },
            _focusVisible: {
              bg: "rgba(255,255,255,0.04)",
              borderColor: "rgba(255,255,255,0.12)",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.12)",
            },
            _dark: {
              bg: "#151b23",
              border: "1px solid rgba(240,246,252,0.14)",
            },
          },
        },
      },
      defaultProps: {
        variant: "filled",
      },
    },
    Badge: {
      baseStyle: {
        borderRadius: "6px",
        px: 2.5,
        py: 0.9,
        textTransform: "none",
        fontWeight: "600",
        borderWidth: "1px",
        bg: "#3b3b3b",
        color: "#f5f1eb",
        borderColor: "rgba(255,255,255,0.12)",
        _dark: {
          bg: "rgba(59,130,246,0.14)",
          color: "#8fd3ff",
          borderColor: "rgba(96,165,250,0.22)",
        },
      },
    },
    Modal: {
      baseStyle: {
        dialog: {
          borderRadius: "8px",
        },
      },
    },
  },
});

export default theme;
