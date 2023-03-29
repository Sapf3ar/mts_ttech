<template>
  <span class="font-changer-title me-1">Шрифт</span>
  <span class="text-nowrap">
    <span
      @click="changeFontSize(1.5)"
      :class="{
        [fzClassesOnActivity]: fZMultiplier === 1.5,
      }"
      class="letter-font-regular px-1 me-1"
      type="button"
      role="button"
      tabindex="0"
      >А</span
    ><span
      @click="changeFontSize(1.75)"
      :class="{ [fzClassesOnActivity]: fZMultiplier === 1.75 }"
      class="letter-font-big px-1 me-1"
      type="button"
      role="button"
      tabindex="0"
      >А</span
    ><span
      @click="changeFontSize(2)"
      :class="{ [fzClassesOnActivity]: fZMultiplier === 2 }"
      class="letter-font-large px-1"
      type="button"
      role="button"
      tabindex="0"
      >А</span
    ></span
  >
</template>

<script>
import { mapState } from "vuex"

export default {
  data() {
    return {
      fzClassesOnActivity: "border-bottom border-4 border-danger",
    }
  },
  computed: {
    ...mapState({
      fZMultiplier: (state) => state.fontSizeMultiplier,
      defaultFonstSizeMultiplier: (state) =>
        state.defaultSettings.fontSizeMultiplier,
    }),
  },
  methods: {
    changeFontSize(multiplier) {
      this.$store.dispatch("changeFontSize", multiplier)
    },
  },
}
</script>
