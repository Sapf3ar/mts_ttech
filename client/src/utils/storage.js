export default {
  get(key) {
    try {
      return JSON.parse(window.localStorage.getItem(key))
    } catch (e) {
      console.warn("Error in getting data from localStorage", e)
      return null
    }
  },

  set(key, data) {
    try {
      window.localStorage.setItem(key, JSON.stringify(data))
    } catch (e) {
      console.warn("Error in setting data to localStorage", e)
    }
  },
}
