<template>
  <div class="position-relative">
    <swiper
      class="swiper"
      :modules="modules"
      :space-between="0"
      :slides-per-view="1"
      :autoplay="{
        delay: 3500,
        waitForTransition: false,
        pauseOnMouseEnter: false,
      }"
      :speed="1000"
      :navigation="navigation"
      :lazy="true"
      :loop="true"
    >
      <swiper-slide v-for="slide of slides" :key="slide.poster"
        ><div
          class="banner"
          :style="{ backgroundImage: 'url(\'' + slide.poster + '\')' }"
        ></div
      ></swiper-slide>
    </swiper>
    <div class="swiper-button-prev" id="banner-arrow-prev"></div>
    <div class="swiper-button-next" id="banner-arrow-next"></div>
  </div>
</template>

<style>
.banner {
  height: 384px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: center;
  background-size: cover;
  -webkit-box-shadow: 0px 0px 94px 15px rgb(0 0 0 / 55%) inset;
  -moz-box-shadow: 0px 0px 94px 15px rgb(0 0 0 / 55%) inset;
  box-shadow: 0px 0px 94px 15px rgb(0 0 0 / 55%) inset;
}
</style>

<script>
import { defineComponent } from "vue"
import { Navigation, Autoplay } from "swiper"
import { Swiper, SwiperSlide } from "swiper/vue"

export default defineComponent({
  name: "swiper-example-loop-group",
  title: "Loop mode with multiple slides per group",
  url: import.meta.url,
  components: {
    Swiper,
    SwiperSlide,
  },
  setup() {
    return {
      modules: [Navigation, Autoplay],
    }
  },
  data() {
    return {
      slides: this.getDummy(),
      navigation: {
        nextEl: "#banner-arrow-next",
        prevEl: "#banner-arrow-prev",
      },
    }
  },
  methods: {
    getDummy() {
      return this.shuffle([
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202212/20221229/8/20221229083647923toq.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202302/20230210/39/20230210093149272ovo.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202303/20230316/83/20230316083335509zhw.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202303/20230307/56/20230307102742989nsh.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202303/20230307/87/20230307102446876q1f.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202303/20230315/62/20230315155741969ccf.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202303/20230301/64/20230301154842669l66.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202303/20230309/54/202303091614595294lr.jpg?x=1800&y=900&ar=keep",
        },
        {
          poster:
            "https://htv-vsc.kion.ru:32121/CPS/images/universal/film/poster/202112/20211228/47/20211228232459818ipz.jpg?x=1800&y=900&ar=keep",
        },
      ])
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
  },
})
</script>
