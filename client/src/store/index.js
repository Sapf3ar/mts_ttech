import { createStore } from "vuex"
import storage from "@/utils/storage"

const htmlEl = document.documentElement
const defaultSettings = {
  fontSizeMultiplier: 1,
  bodyBackground: "#041423",
  font: "Montserrat",
  letterSpacing: "0px",
}

export default createStore({
  state: {
    fontSizeMultiplier: storage.get("fontSizeMultiplier"),
    bodyBackground: storage.get("bodyBackground"),
    isVImode: storage.get("isVImode"),
    font: storage.get("font") || defaultSettings.font,
    letterSpacing:
      storage.get("letterSpacing") || defaultSettings.letterSpacing,

    currentMovieId: null,
    movieIsPlaying: false,
    defaultSettings,
  },
  getters: {
    idDarkTheme(state) {
      return state.bodyBackground !== "#ffffff"
    },
  },
  mutations: {
    CHANGE_VI_MODE(state, isVImode) {
      state.isVImode = isVImode
    },
    CHANGE_FONT_SIZE_MULTIPLIER(state, multiplier) {
      state.fontSizeMultiplier = multiplier
    },
    CHANGE_BACKGROUND(state, color) {
      state.bodyBackground = color
    },
    SET_CURRENT_MOVIE_ID(state, movieId) {
      state.currentMovieId = movieId
    },
    SET_FONT(state, font) {
      state.font = font
    },
    SET_LETTER_SPACING(state, letterSpacing) {
      state.letterSpacing = letterSpacing
    },
  },
  actions: {
    switchVIVisibility({ commit, state, dispatch }) {
      const value = state.isVImode
      const newValue = !value

      const fz = newValue ? 1.5 : 1
      dispatch("changeFontSize", fz)

      commit("CHANGE_VI_MODE", newValue)
      storage.set("isVImode", newValue)
    },
    changeFontSize({ commit, dispatch }, payload) {
      commit("CHANGE_FONT_SIZE_MULTIPLIER", payload)
      dispatch("syncVisibleSettings")
      storage.set("fontSizeMultiplier", payload)
    },
    changeBodyBackground({ commit, dispatch }, payload) {
      commit("CHANGE_BACKGROUND", payload)
      dispatch("syncVisibleSettings")
      storage.set("bodyBackground", payload)
    },
    setFont({ commit, dispatch }, payload) {
      commit("SET_FONT", payload)
      dispatch("syncVisibleSettings")
      storage.set("font", payload)
    },
    setLetterSpacing({ commit, dispatch }, payload) {
      commit("SET_LETTER_SPACING", payload)
      dispatch("syncVisibleSettings")
      storage.set("letterSpacing", payload)
    },
    syncVisibleSettings({ state, getters }) {
      htmlEl.style.setProperty("--tt-fz-multiplier", state.fontSizeMultiplier)
      htmlEl.style.setProperty("--tt-body-bg", state.bodyBackground)
      htmlEl.style.setProperty("--tt-ffamily", state.font)
      htmlEl.style.setProperty("--tt-letter-spacing", state.letterSpacing)

      if (getters.idDarkTheme) {
        htmlEl.dataset.bsTheme = "dark"
      } else {
        htmlEl.dataset.bsTheme = "light"
      }

      let fontWeight = 400
      if (state.fontSizeMultiplier === 1.75) {
        fontWeight = 600
      } else if (state.fontSizeMultiplier === 2) {
        fontWeight = 800
      }

      htmlEl.style.setProperty("--tt-fw", fontWeight)
    },
    resetSettings({ state, dispatch }) {
      for (const key in state.defaultSettings) {
        const value = state.defaultSettings[key]

        state[key] = value
        storage.set(key, value)
      }
      dispatch("syncVisibleSettings")
    },
  },
  modules: {},
})
