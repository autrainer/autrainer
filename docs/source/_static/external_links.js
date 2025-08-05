$(function () {
  $("a.external").each(function () {
    const u = new URL(this.href, location.origin);
    if (
      u.hostname !== "autrainer.github.io" ||
      !u.pathname.startsWith("/autrainer")
    )
      $(this).attr({ target: "_blank", rel: "noopener noreferrer" });
  });
});
