$(document).ready(function () {
  $("#cli-reference .highlight-aucli.notranslate").each(function () {
    var defaultHighlight = $(this)
      .siblings(".highlight-default.notranslate")
      .first();

    if (defaultHighlight.length) {
      defaultHighlight.html($(this).html());
      $(this).remove();
    }
  });
});
