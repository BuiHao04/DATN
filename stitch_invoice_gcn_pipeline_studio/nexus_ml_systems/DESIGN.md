---
name: Nexus ML Systems
colors:
  surface: '#fbf8ff'
  surface-dim: '#dbd9e2'
  surface-bright: '#fbf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f4f2fc'
  surface-container: '#efedf6'
  surface-container-high: '#e9e7f0'
  surface-container-highest: '#e3e1ea'
  on-surface: '#1a1b22'
  on-surface-variant: '#454652'
  inverse-surface: '#2f3037'
  inverse-on-surface: '#f2eff9'
  outline: '#757684'
  outline-variant: '#c5c5d4'
  surface-tint: '#4355b9'
  primary: '#24389c'
  on-primary: '#ffffff'
  primary-container: '#3f51b5'
  on-primary-container: '#cacfff'
  inverse-primary: '#bac3ff'
  secondary: '#505f76'
  on-secondary: '#ffffff'
  secondary-container: '#d0e1fb'
  on-secondary-container: '#54647a'
  tertiary: '#6c3400'
  on-tertiary: '#ffffff'
  tertiary-container: '#8f4700'
  on-tertiary-container: '#ffc7a2'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dee0ff'
  primary-fixed-dim: '#bac3ff'
  on-primary-fixed: '#00105c'
  on-primary-fixed-variant: '#293ca0'
  secondary-fixed: '#d3e4fe'
  secondary-fixed-dim: '#b7c8e1'
  on-secondary-fixed: '#0b1c30'
  on-secondary-fixed-variant: '#38485d'
  tertiary-fixed: '#ffdcc6'
  tertiary-fixed-dim: '#ffb784'
  on-tertiary-fixed: '#301400'
  on-tertiary-fixed-variant: '#713700'
  background: '#fbf8ff'
  on-background: '#1a1b22'
  surface-variant: '#e3e1ea'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  title-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
  mono-data:
    fontFamily: jetbrainsMono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  sidebar-width: 260px
  container-gap: 1.5rem
  table-cell-padding: 0.75rem 1rem
  stack-gap-sm: 0.5rem
  stack-gap-md: 1rem
---

## Brand & Style
The design system is engineered for high-density ML workflows, focusing on clarity, precision, and technical rigor. It targets ML engineers and data scientists who require immediate access to pipeline metrics, training logs, and model performance data without visual fatigue. 

The aesthetic is **Corporate Modern with a Minimalist lean**, emphasizing a systematic approach to data visualization. It prioritizes functionality over decoration, using ample white space to separate complex data sets while maintaining a high information density suitable for expert users. The interface should feel like a high-performance instrument: reliable, responsive, and unobtrusive.

## Colors
The palette is anchored by a **Deep Indigo** primary, providing a professional and authoritative foundation for an engineering environment. 

- **Primary (#3F51B5):** Used for primary actions, active navigation states, and key progress indicators.
- **Neutral/Secondary (#64748B):** A muted slate for secondary text and icons, preventing visual competition with primary data.
- **Surface & Background:** The background uses a cool light gray (#F8FAFC) to define the workspace, while white (#FFFFFF) surfaces are used for cards and data tables to create a clear "layering" effect.
- **Semantic Palette:** High-saturation tokens for Success, Warning, Error, and Info are used strictly for status indicators and pipeline health to ensure immediate recognition of system states.

## Typography
This design system utilizes **Inter** for its neutral, highly legible characteristic at small sizes—critical for OCR and GCN data validation. 

- **Scale:** A compact scale is used to maximize vertical real estate. 
- **Data Display:** For model logs, JSON outputs, or hex codes, the system incorporates **JetBrains Mono** to ensure character differentiation (e.g., 0 vs O).
- **Labels:** Uppercase labels with slight tracking are used for table headers and section dividers to establish a clear hierarchy without increasing font size.

## Layout & Spacing
The layout follows a **Fixed-Fluid hybrid model**. A fixed left-hand sidebar (260px) provides persistent navigation, while the main content area utilizes a fluid 12-column grid.

- **Grid:** 1.5rem (24px) gutters between main containers.
- **Density:** We utilize a "Compact" density model for data tables and lists, minimizing padding to 12px (0.75rem) to allow more rows to be visible above the fold.
- **Breakpoints:** 
  - Mobile (<768px): Sidebar collapses to a drawer; 1-column layout.
  - Tablet (768px-1200px): Sidebar icons only; 2-column dashboard layout.
  - Desktop (>1200px): Full sidebar; multi-column data views.

## Elevation & Depth
Depth is achieved through **Tonal Layering and Soft Shadows** rather than heavy borders.

- **Level 0 (Background):** #F8FAFC (Canvas).
- **Level 1 (Cards/Tables):** White surface with a 1px border (#E2E8F0) and an ultra-soft ambient shadow (0px 1px 3px rgba(0,0,0,0.05)).
- **Level 2 (Modals/Popovers):** White surface with a more pronounced shadow (0px 10px 15px -3px rgba(0,0,0,0.1)) to indicate focus.
- **Interaction:** On-hover states for interactive table rows should use a subtle tint change (#F1F5F9) rather than an elevation increase to maintain layout stability.

## Shapes
The design system utilizes a **Soft (4px/0.25rem)** rounding strategy. This provides a modern feel while retaining a structured, precise look suitable for an engineering tool. 

- **Standard Elements:** 4px radius for buttons, inputs, and cards.
- **Interactive Small Elements:** Chips and status indicators may use a slightly higher radius (8px) to differentiate them from functional inputs.
- **Data Points:** Graph markers and nodes in GCN visualizations remain sharp or use minimal 2px rounding to ensure technical accuracy.

## Components
- **Data Tables:** High-density rows with `sticky` headers. Include inline sparklines for epoch-over-epoch loss monitoring. Row height set to 40px.
- **Status Chips:** Small, rounded containers using light semantic backgrounds with high-contrast text. Example: `Success` uses light green background with dark green text and a leading 12px check icon.
- **Progress Bars:** Thin (4px height) linear bars. Use the primary indigo for standard processes and semantic colors for specific pipeline stages (e.g., Error red if a stage fails).
- **Fixed Sidebar:** Dark-themed or high-contrast slate sidebar with clear icon-label pairings. Active state indicated by a 4px primary-colored left border.
- **Input Fields:** Outlined style with 1px border (#CBD5E1). Focus state uses 1px primary border with a 2px primary glow (low opacity).
- **Monitors:** Large card containers for GCN graph visualizations or OCR preview windows, using a dark gray inner-well (#1E293B) for high-contrast image review.