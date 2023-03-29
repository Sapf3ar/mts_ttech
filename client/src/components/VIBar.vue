<style src="@/scss/vi-bar.scss" lang="scss" />
<template>
  <div
    class="container-fluid bg-light"
    data-bs-theme="light"
    @keydown.enter="silance"
  >
    <div class="container-lg">
      <div
        class="row align-items-center vi-container text-md-start text-center"
      >
        <div class="col-md-6 col-lg-3 my-md-0 my-2 font-changer-container">
          <FontSizeChangerContainer />
        </div>
        <div
          class="col-md-6 col-lg-3 my-md-0 my-2 text-md-end text-lg-start bg-changer-container"
        >
          <BackgroundChangerContainer />
        </div>
        <div class="col-md-6 col-lg-3 my-md-0 my-2">
          <div class="dropdown">
            <span
              class="dropdown-toggle"
              type="button"
              role="button"
              tabindex="0"
              data-bs-toggle="dropdown"
              aria-expanded="false"
            >
              Настройки
            </span>
            <FontSettings />
          </div>
        </div>
        <div
          @click="resetVISettings"
          class="col-md-6 col-lg-3 my-md-0 my-2 text-md-end text-lg-start"
          type="button"
          role="button"
          tabindex="0"
        >
          Обычная версия
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapState } from "vuex"
import FontSettings from "@/components/FontSettings"
import BackgroundChangerContainer from "@/components/BackgroundChangerContainer"
import FontSizeChangerContainer from "@/components/FontSizeChangerContainer"

export default {
  components: {
    FontSettings,
    BackgroundChangerContainer,
    FontSizeChangerContainer,
  },
  data() {
    return {}
  },
  computed: {
    ...mapState({}),
  },
  methods: {
    resetVISettings() {
      this.$store.dispatch("resetSettings")
      this.$store.dispatch("switchVIVisibility")
    },
    silance(event) {
      if (event.target) {
        event.target.click()
      }
    },
  },
}
</script>
