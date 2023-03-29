<template>
  <button
    type="button"
    role="button"
    tabindex="0"
    class="btn hover-button"
    @keypress.enter="switchVIVisibility"
  >
    <img :src="icon_glasses_white" alt="icon_glasses_white" />
  </button>
  <div class="hover-container position-absolute" type="button" role="button">
    <button type="button" class="btn" @click="switchVIVisibility">
      <b class="text-black me-4">Версия для слабовидящих</b>
      <img :src="icon_glasses_black" alt="icon_glasses_black" />
    </button>
  </div>
</template>

<style scoped>
.hover-container {
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.5s;
  right: 12px;
  top: 0px;
}
.hover-container:hover {
  pointer-events: auto;
  opacity: 1;
}
.hover-container .btn {
  background: #ffffff;
  border-radius: 30px;
  font-size: 20px;
  white-space: nowrap;
}

.hover-button:focus-within {
  outline: none;
}
.hover-button:focus-within + .hover-container,
.hover-button:hover + .hover-container {
  opacity: 1;
  pointer-events: auto;
}
</style>

<script>
import icon_glasses_black from "@/assets/icon_glasses_black.svg"
import icon_glasses_white from "@/assets/icon_glasses_white.svg"
export default {
  name: "VIButton",
  data() {
    return { icon_glasses_black, icon_glasses_white }
  },
  methods: {
    switchVIVisibility() {
      this.$store.dispatch("switchVIVisibility")
    },
  },
}
</script>
