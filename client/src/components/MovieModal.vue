<template>
  <div
    class="modal fade"
    id="movieModal"
    tabindex="-1"
    aria-labelledby="movieModalLabel"
    aria-hidden="true"
  >
    <div
      class="modal-dialog modal-xl modal-fullscreen-md-down modal-dialog-centered modal-dialog-scrollable"
    >
      <div class="modal-content">
        <button
          type="button"
          class="btn-close"
          data-bs-dismiss="modal"
          aria-label="Close"
        ></button>
        <div class="modal-body">
          <MovieDetails v-if="currentMovieId !== null" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.btn-close {
  position: absolute;
  left: 25px;
  top: 25px;
  z-index: 10;
}

@media (max-width: 768px) {
  .btn-close {
    left: unset;
    right: 25px;
  }
}

@media (min-width: 768px) {
  .modal-body {
    padding-left: 50px;
  }
}
</style>

<script>
import { mapState } from "vuex"

import MovieDetails from "@/components/MovieDetails"

export default {
  components: {
    MovieDetails,
  },
  data() {
    return {}
  },
  computed: {
    ...mapState({
      currentMovieId: (state) => state.currentMovieId,
    }),
  },
}
</script>
