<template>
  <svg
    width="40"
    height="38"
    viewBox="0 0 48 42"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <g filter="url(#filter0_d_0_318)">
      <path
        d="M11.65 38L14.9 23.95L4 14.5L18.4 13.25L24 0L29.6 13.25L44 14.5L33.1 23.95L36.35 38L24 30.55L11.65 38Z"
        :fill="bgColor"
      />
    </g>
    <defs>
      <filter
        id="filter0_d_0_318"
        x="0"
        y="0"
        width="48"
        height="46"
        filterUnits="userSpaceOnUse"
        color-interpolation-filters="sRGB"
      >
        <feFlood flood-opacity="0" result="BackgroundImageFix" />
        <feColorMatrix
          in="SourceAlpha"
          type="matrix"
          values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
          result="hardAlpha"
        />
        <feOffset dy="4" />
        <feGaussianBlur stdDeviation="2" />
        <feComposite in2="hardAlpha" operator="out" />
        <feColorMatrix
          type="matrix"
          values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0"
        />
        <feBlend
          mode="normal"
          in2="BackgroundImageFix"
          result="effect1_dropShadow_0_318"
        />
        <feBlend
          mode="normal"
          in="SourceGraphic"
          in2="effect1_dropShadow_0_318"
          result="shape"
        />
      </filter>
    </defs>
  </svg>
</template>

<script>
import { mapGetters } from "vuex"

export default {
  data() {
    return {}
  },
  computed: {
    bgColor() {
      return this.isDarkTheme ? "#ffffff" : "#000000"
    },
    ...mapGetters({
      isDarkTheme: "idDarkTheme",
    }),
  },
}
</script>
