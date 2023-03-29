<template>
  <div class="home">
    <SliderBanner class="my-5" v-if="!isVImode" />
    <SliderPopular class="mb-5" :isVImode="isVImode" />
    <VideoPlayer
      class="position-fixed w-100 video-player-container"
      v-if="movieIsPlaying"
    />
  </div>
</template>

<style scoped>
.video-player-container {
  z-index: 2000;
  top: 0;
  left: 0;
  height: 100%;
}
.video-player-container .video-js {
  height: 100vh;
}
</style>

<script>
import { mapState } from "vuex"

import VideoPlayer from "@/components/VideoPlayer.vue"
import SliderBanner from "@/components/SliderBanner.vue"
import SliderPopular from "@/components/SliderPopular.vue"

export default {
  name: "VideoExample",
  components: {
    VideoPlayer,
    SliderPopular,
    SliderBanner,
  },
  data() {
    return {}
  },
  computed: {
    ...mapState({
      isVImode: (state) => state.isVImode,
      movieIsPlaying: (state) => state.movieIsPlaying,
    }),
  },
}
</script>
