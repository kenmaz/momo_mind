$(function() {
  var myDropzone = new Dropzone($("#stage")[0],{
    url: "/upload",
    parallelUploads: 1,
    maxThumbnailFilesize: 1,
    maxFilesize: 1,
    uploadMultiple: false,
    thumbnailWidth: 400,
    thumbnailHeight: 400,
    maxFiles: 1,
    previewTemplate:
      '<div class="dz-preview dz-file-preview">'+
        '<div class="dz-details text-center"><img data-dz-thumbnail width="100%" style="margin-top: 10px"/></div>'+
        '<div class="dz-progress"><span class="dz-upload" data-dz-uploadprogress></span></div>'+
        '<div class="dz-error-message"><span data-dz-errormessage></span></div>'+
      '</div>',
    accept: function(file, done) {
      done();
    },
    init: function() {
      var statusLabel = $("#status");

      this.on("addedfile", function() {
        if (this.files[1] != null) {
          this.removeFile(this.files[0]);
        }
        $("#drag_msg").hide();
        $("#status").show();
        $("#stage").css("background-color", "white");
      });
      this.on("processing", function(file) {
        console.log("processing");
        $("#status").text('processing')

        var table = $("#results");
        table.empty();
      })
      this.on("uploadprogress", function(file, progress, bytesSent) {
        console.log("uploadprogress");
        $("#status").text('uploadprogress:'+progress);
        if (progress == 100) {
          $("#status").text('prosessing...');
        }
      })
      this.on("sending", function(file) {
        console.log("sending");
      })
      this.on("success", function(file, response) {
        $("#status").text('success')
        console.log("success");
        console.log(response);

        var table = $("#results");

        var results = response['results'];
        if (results.length == 0) {
          $("#status").text('顔を検知出来ませんでした')

        } else {
          $(results).each(function(idx, item) {
            console.log(item);
            var row = $("<div class='row'>");
            $("#results").append(row);

            var col_img = $("<div class='img pull-left'>").append($("<img>").attr('src', item['file']));
            row.append(col_img);

            var col_names = $("<div class='names'>");
            row.append(col_names);

            var ul = $("<table class='rank'>");
            col_names.append(ul);

            $(item['rank']).each(function(j, member) {
              var li = $("<tr><td><span class='"+member['name_ascii']+" name'>"+member["name"]+"</span></td><td>"+member["rate"]+"%</td></tr>");
              ul.append(li);
            });

            row.append($("<div class='clearfix'>"));
          });
        }
      })
      this.on("error", function(file) {
        $("#status").text('error')
        console.log("error");
      })
    }
  });
});
