<template>
  <span class="bg-changer-title me-1">Фон</span>
  <span class="text-nowrap">
    <span
      @click="changeBodyBackground('#ffffff')"
      :class="{
        [backgroundClassesOnActivity]: bodyBackground === '#ffffff',
      }"
      class="letter-bg-regular px-1 me-1"
      type="button"
      role="button"
      tabindex="0"
      >Ф</span
    ><span
      @click="changeBodyBackground('#000000')"
      :class="{
        [backgroundClassesOnActivity]: bodyBackground === '#000000',
      }"
      class="letter-bg-black px-1 me-1"
      type="button"
      role="button"
      tabindex="0"
      >Ф</span
    ><span
      @click="changeBodyBackground(this.defaultBodyBackgroundColor)"
      :class="{
        [backgroundClassesOnActivity]:
          bodyBackground === defaultBodyBackgroundColor,
      }"
      class="letter-bg-dark px-1 ms-1"
      type="button"
      role="button"
      tabindex="0"
      >Ф</span
    ></span
  >
</template>

<script>
import { mapState } from "vuex"

export default {
  data() {
    return {
      backgroundClassesOnActivity: "border border-4 border-danger",
    }
  },
  computed: {
    ...mapState({
      bodyBackground: (state) => state.bodyBackground,
      defaultBodyBackgroundColor: (state) =>
        state.defaultSettings.bodyBackground,
    }),
  },
  methods: {
    changeBodyBackground(color) {
      this.$store.dispatch("changeBodyBackground", color)
    },
  },
}
</script>
