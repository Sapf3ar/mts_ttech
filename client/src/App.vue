<style lang="scss" src="@/scss/app.scss" />
<template>
  <div>
    <VIBar v-if="isVImode" />
    <div class="container-lg">
      <NavHeader />
      <router-view />
    </div>
    <MovieModal />
  </div>
</template>

<script>
import { mapState } from "vuex"

import NavHeader from "@/components/NavHeader.vue"
import VIBar from "@/components/VIBar.vue"
import MovieModal from "@/components/MovieModal.vue"

export default {
  components: {
    NavHeader,
    VIBar,
    MovieModal,
  },

  computed: {
    ...mapState({
      isVImode: (state) => state.isVImode,
    }),
  },
  mounted() {
    this.$store.dispatch("syncVisibleSettings")
  },
}
</script>
