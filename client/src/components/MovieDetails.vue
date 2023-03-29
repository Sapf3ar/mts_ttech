<template>
  <div class="row">
    <div
      class="d-flex flex-column col-md-6 order-1 order-md-0"
      :class="isVImode ? 'mt-3' : 'mt-md-5 mt-3'"
    >
      <div class="row align-items-center mb-1">
        <div class="col-auto"><IconStar /><b class="ms-1">9.2</b></div>
        <div class="col">
          <h3 class="m-0">{{ movie.title }}</h3>
        </div>
      </div>
      <div class="col" :class="isVImode ? 'my-2' : 'mt-md-5 my-3'">
        {{ movie.description }}
      </div>
      <div class="col-auto">
        <button
          type="button"
          role="button"
          class="btn w-100"
          :class="movieIsAvailable ? 'btn-danger' : 'btn-secondary disabled'"
          @click="playMovie"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="48"
            height="48"
            viewBox="0 0 48 48"
            fill="none"
            class="mx-2"
          >
            <path d="M9 43V5L39 24L9 43Z" fill="white" />
          </svg>
          <span
            class="d-sm-inline d-none text-nowrap watch-btn text-uppercase mx-2"
            >Смотреть фильм</span
          >
        </button>
      </div>
    </div>
    <div class="col-md-6 col-5">
      <img
        class="w-100"
        style="min-width: 250px"
        :src="movie.poster"
        :alt="movie.title"
      />
    </div>
  </div>
</template>

<style scoped>
.watch-btn {
  font-weight: bold;
}
</style>

<script>
import { mapState } from "vuex"
import IconStar from "./IconStar"

import dummyMovies from "@/data/movies"

export default {
  data() {
    console.log(dummyMovies, this.currentMovieId)
    return {
      isPlaying: false,
      movieIsAvailable: true,
    }
  },
  computed: {
    ...mapState({
      currentMovieId: (state) => state.currentMovieId,
      isVImode: (state) => state.isVImode,
    }),
    movie() {
      return dummyMovies.find((movie) => movie.id === this.currentMovieId)
    },
  },
  methods: {
    playMovie() {
      this.$store.state.movieIsPlaying = true
    },
  },
  components: { IconStar },
}
</script>
