// Chart.js Helper Functions and Utilities

/**
 * Global Chart configuration settings
 */
const chartConfig = {
  defaultFont: {
    family:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
    size: 13,
    weight: 500,
  },
  colors: {
    primary: "#0ea5e9",
    primaryDark: "#1e3a8a",
    success: "#10b981",
    danger: "#ef4444",
    warning: "#f59e0b",
    info: "#3b82f6",
    gray: "#9ca3af",
    grayLight: "#e5e7eb",
    white: "#ffffff",
  },
};

/**
 * Configure Chart.js defaults
 */
function configureChartDefaults() {
  Chart.defaults.font.family = chartConfig.defaultFont.family;
  Chart.defaults.font.size = chartConfig.defaultFont.size;
  Chart.defaults.font.weight = chartConfig.defaultFont.weight;
  Chart.defaults.color = "#6b7280";
  Chart.defaults.borderColor = "#e5e7eb";
}

/**
 * Get a formatted percentage string
 */
function formatPercentage(value, total) {
  if (total === 0) return "0%";
  return ((value / total) * 100).toFixed(1) + "%";
}

/**
 * Format numbers with thousand separators
 */
function formatNumber(num) {
  if (typeof num !== "number") return num;
  return num.toLocaleString("id-ID");
}

/**
 * Create a gradient color array
 */
function createGradient(ctx, colors) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 400);
  colors.forEach((color, index) => {
    gradient.addColorStop(index / (colors.length - 1), color);
  });
  return gradient;
}

/**
 * Common chart options for consistency
 */
const commonChartOptions = {
  responsive: true,
  maintainAspectRatio: true,
  layout: {
    padding: 0,
  },
  plugins: {
    legend: {
      labels: {
        padding: 15,
        usePointStyle: true,
        font: {
          size: 12,
          weight: "500",
        },
      },
    },
    tooltip: {
      backgroundColor: "rgba(0, 0, 0, 0.8)",
      padding: 12,
      titleFont: {
        size: 13,
        weight: "bold",
      },
      bodyFont: {
        size: 12,
      },
      borderColor: "#e5e7eb",
      borderWidth: 1,
      displayColors: true,
      callbacks: {
        labelColor: function (context) {
          return {
            borderColor: context.dataset.backgroundColor,
            backgroundColor: context.dataset.backgroundColor,
          };
        },
      },
    },
  },
};

/**
 * Initialize Chart.js with default configuration
 */
function initChartJS() {
  configureChartDefaults();
}

/**
 * Convert chart instance to image data
 */
function exportChartAsImage(chartInstance, fileName) {
  const image = chartInstance.toBase64Image();
  const link = document.createElement("a");
  link.href = image;
  link.download = fileName || "chart.png";
  link.click();
}

/**
 * Destroy all chart instances on a page
 */
function destroyAllCharts() {
  Chart.helpers.each(Chart.instances, function (instance) {
    instance.destroy();
  });
}

/**
 * Format currency values
 */
function formatCurrency(value, currency = "IDR") {
  if (typeof value !== "number") return value;
  return value.toLocaleString("id-ID", {
    style: "currency",
    currency: currency,
  });
}

/**
 * Debounce function for resize events
 */
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Add event listeners for responsive charts
 */
function setupResponsiveCharts() {
  window.addEventListener(
    "resize",
    debounce(function () {
      // Trigger chart redraw if needed
      Chart.instances.forEach((instance) => {
        instance.resize();
      });
    }, 250),
  );
}

// Initialize on document ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initChartJS);
} else {
  initChartJS();
}

// Setup responsive charts
setupResponsiveCharts();
