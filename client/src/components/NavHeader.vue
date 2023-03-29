<template>
  <nav
    class="navbar navbar-expand-lg row justify-content-between align-items-start"
    :class="{
      'align-items-lg-end my-2': isVImode,
      'align-items-lg-center': !isVImode,
    }"
  >
    <LogoIcon />

    <div class="col">
      <div class="row align-items-center justify-content-end">
        <div class="col-auto col-lg text-end">
          <button
            class="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarToggler"
            aria-controls="navbarToggler"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>

          <div id="navbarToggler" :class="collapsClasses">
            <ul
              class="navbar-nav justify-content-end w-100"
              :class="isVImode ? '' : 'me-2'"
            >
              <li class="nav-item">
                <a class="nav-link active" aria-current="page" href="#"
                  >Главная</a
                >
              </li>
              <li class="nav-item">
                <a class="nav-link disabled">Фильмы</a>
              </li>
              <li class="nav-item">
                <a class="nav-link disabled">Сериалы</a>
              </li>
              <li class="nav-item d-lg-none" v-if="!isVImode">
                <button
                  @click="switchVIVisibility"
                  class="btn text-uppercase pe-0 mb-3"
                  style="font-weight: bold"
                >
                  Версия для слабовидящих
                </button>
              </li>
            </ul>
            <SearchBar :class="searchBarClasses" />
            <PersonButton v-if="false" />
          </div>
        </div>
        <div
          class="col-auto text-end position-relative d-lg-block d-none"
          v-if="!isVImode"
        >
          <VIButton />
        </div>
      </div>
    </div>
  </nav>
</template>

<style scoped>
.search-form-container {
  max-width: 290px;
}
</style>

<script>
import { mapState } from "vuex"

import LogoIcon from "@/components/LogoIcon"
import SearchBar from "@/components/SearchBar"
import PersonButton from "@/components/PersonButton"
import VIButton from "@/components/VIButton"

export default {
  components: {
    LogoIcon,
    PersonButton,
    VIButton,
    SearchBar,
  },
  data() {
    return {}
  },
  computed: {
    ...mapState({
      isVImode: (state) => state.isVImode,
      VIButtonClassesOnVI() {
        return "orde"
      },
      collapsClasses() {
        return {
          "flex-wrap justify-content-between": this.isVImode,
          "collapse navbar-collapse": true,
        }
      },
      searchBarClasses() {
        return {
          "order-1 ": this.isVImode,
          "search-form-container": !this.isVImode,
          "w-100": true,
        }
      },
    }),
  },
  methods: {
    switchVIVisibility() {
      this.$store.dispatch("switchVIVisibility")
    },
  },
}
</script>
