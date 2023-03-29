<template>
  <div>
    <h3 v-if="!isVImode">Популярно сейчас</h3>
    <div class="position-relative">
      <swiper
        class="swiper tt-swiper"
        :modules="modules"
        :breakpoints="breakpoints"
        :navigation="navigation"
        :keyboard="{
          enabled: true,
          onlyInViewport: true,
          pageUpDown: true,
        }"
        @slideChange="onSlideChange"
        :key="JSON.stringify(breakpoints)"
      >
        <swiper-slide
          v-for="slide of slides"
          class="p-1"
          :key="slide.id"
          @click="openModal(slide.id)"
          ><button class="btn p-0 w-100" role="button">
            <img
              :src="slide.poster"
              :alt="slide.title"
              class="w-100"
              :style="{
                'border-radius': isVImode ? '30px' : '10px',
              }"
            /></button
        ></swiper-slide>
      </swiper>

      <div
        class="swiper-button-prev"
        :style="{ color: isDarkTheme ? 'initial' : 'black' }"
        id="popular-arrow-prev"
      ></div>
      <div
        class="swiper-button-next"
        :style="{ color: isDarkTheme ? 'initial' : 'black' }"
        id="popular-arrow-next"
      ></div>
    </div>
  </div>
</template>

<style>
.tt-swiper {
  margin-left: -0.25rem !important;
  margin-right: -0.25rem !important;
}
</style>

<script>
import { Navigation, Keyboard } from "swiper"
import { Swiper, SwiperSlide } from "swiper/vue"
import { mapGetters, mapState } from "vuex"
import { Modal } from "bootstrap"

import dummyMovies from "@/data/movies"

export default {
  name: "swiper-example-loop-group",
  title: "Loop mode with multiple slides per group",
  url: import.meta.url,
  components: {
    Swiper,
    SwiperSlide,
  },
  setup() {
    return {
      modules: [Navigation, Keyboard],
    }
  },
  props: {
    isVImode: { type: Boolean, default: false },
  },
  data() {
    return {
      slides: this.getDummy(),
      navigation: {
        nextEl: "#popular-arrow-next",
        prevEl: "#popular-arrow-prev",
      },
      modal: null,
    }
  },
  computed: {
    ...mapState({
      movieIsPlaying: (state) => state.movieIsPlaying,
    }),
    ...mapGetters({
      isDarkTheme: "idDarkTheme",
    }),
    breakpoints() {
      const isVI = this.isVImode
      return {
        0: {
          slidesPerView: 2,
          spaceBetween: 10,
        },
        576: {
          slidesPerView: isVI ? 2 : 3,
          spaceBetween: 20,
        },
        768: {
          slidesPerView: isVI ? 2 : 4,
          spaceBetween: 30,
        },
        992: {
          slidesPerView: isVI ? 2 : 5,
          spaceBetween: 40,
        },
      }
    },
  },
  methods: {
    onSlideChange() {
      // todo
      // swiper.slides[swiper.activeIndex].focus()
    },
    getDummy() {
      return this.shuffle(dummyMovies)
    },
    shuffle(array) {
      let currentIndex = array.length,
        randomIndex

      // While there remain elements to shuffle.
      while (currentIndex != 0) {
        // Pick a remaining element.
        randomIndex = Math.floor(Math.random() * currentIndex)
        currentIndex--

        // And swap it with the current element.
        ;[array[currentIndex], array[randomIndex]] = [
          array[randomIndex],
          array[currentIndex],
        ]
      }

      return array
    },
    openModal(movieId) {
      const modalEl = document.getElementById("movieModal")
      this.modal = new Modal(modalEl, {})
      this.modal.show()

      this.$store.commit("SET_CURRENT_MOVIE_ID", movieId)
    },
  },
  watch: {
    movieIsPlaying(isPlaying) {
      if (isPlaying) {
        document.body.classList.add("overflow-hidden")
        this.modal.hide()
      } else {
        document.body.classList.remove("overflow-hidden")
        this.modal.show()
      }
    },
  },
}
</script>
