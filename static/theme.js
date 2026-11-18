/* Applies saved appearance settings before first paint. Settings: mode (light|dark|system), accent. */
(function () {
  var root = document.documentElement, mq = window.matchMedia("(prefers-color-scheme: dark)");
  function get(k, d) { try { return localStorage.getItem(k) || d; } catch (e) { return d; } }
  function apply() {
    var m = get("mode", "light");
    root.dataset.mode = (m === "dark" || (m === "system" && mq.matches)) ? "dark" : "light";
    root.dataset.accent = get("accent", "violet");
  }
  apply();
  if (mq.addEventListener) mq.addEventListener("change", apply);
  window.Theme = {
    get: get,
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} apply(); },
    reset: function () { try { localStorage.removeItem("mode"); localStorage.removeItem("accent"); } catch (e) {} apply(); }
  };
})();
