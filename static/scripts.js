(function () {
    "use strict";

    // -----------------------------
    // Get HTML elements
    // -----------------------------

    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const dropzone = document.getElementById("dropzone-label");
    const fileList = document.getElementById("file-list");
    const uploadBtn = document.getElementById("upload-btn");

    const askForm = document.getElementById("ask-form");
    const questionInput = document.getElementById("question");
    const askBtn = document.getElementById("ask-btn");

    const suggestions = document.querySelectorAll(".suggestion");


    // -----------------------------
    // Store selected files
    // -----------------------------

    let stagedFiles = [];


    // -----------------------------
    // Format file size
    // -----------------------------

    function formatBytes(bytes) {
        if (!bytes) {
            return "0 B";
        }

        if (bytes < 1024) {
            return bytes + " B";
        }

        if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + " KB";
        }

        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }


    // -----------------------------
    // Sync files with input
    // -----------------------------

    function syncFileInput() {
        const dataTransfer = new DataTransfer();

        stagedFiles.forEach(function (file) {
            dataTransfer.items.add(file);
        });

        fileInput.files = dataTransfer.files;
    }


    // -----------------------------
    // Display selected files
    // -----------------------------

    function renderFiles() {

        fileList.innerHTML = "";


        if (stagedFiles.length === 0) {
            fileList.hidden = true;
            uploadBtn.disabled = true;
            return;
        }


        stagedFiles.forEach(function (file, index) {

            const item = document.createElement("li");

            const name = document.createElement("span");
            name.className = "name";
            name.textContent = file.name;
            name.title = file.name;


            const size = document.createElement("span");
            size.textContent = formatBytes(file.size);
            size.style.color = "var(--text-muted)";
            size.style.fontSize = "10px";


            const removeButton =
                document.createElement("button");

            removeButton.type = "button";
            removeButton.textContent = "Remove";


            removeButton.addEventListener(
                "click",
                function () {

                    stagedFiles.splice(index, 1);

                    syncFileInput();

                    renderFiles();
                }
            );


            item.appendChild(name);
            item.appendChild(size);
            item.appendChild(removeButton);

            fileList.appendChild(item);
        });


        fileList.hidden = false;

        uploadBtn.disabled = false;
    }


    // -----------------------------
    // Add PDF files
    // -----------------------------

    function addFiles(files) {

        if (!files) {
            return;
        }


        Array.from(files).forEach(function (file) {

            const isPDF =
                file.name.toLowerCase().endsWith(".pdf");


            if (!isPDF) {
                return;
            }


            const alreadyExists =
                stagedFiles.some(function (existingFile) {

                    return (
                        existingFile.name === file.name &&
                        existingFile.size === file.size
                    );

                });


            if (!alreadyExists) {
                stagedFiles.push(file);
            }

        });


        syncFileInput();

        renderFiles();
    }


    // -----------------------------
    // Browse files
    // -----------------------------

    if (fileInput) {

        fileInput.addEventListener(
            "change",
            function () {

                addFiles(fileInput.files);

            }
        );

    }


    // -----------------------------
    // Drag and Drop
    // -----------------------------

    if (dropzone) {

        dropzone.addEventListener(
            "dragenter",
            function (event) {

                event.preventDefault();

                dropzone.classList.add("is-dragover");

            }
        );


        dropzone.addEventListener(
            "dragover",
            function (event) {

                event.preventDefault();

                dropzone.classList.add("is-dragover");

            }
        );


        dropzone.addEventListener(
            "dragleave",
            function (event) {

                event.preventDefault();

                dropzone.classList.remove("is-dragover");

            }
        );


        dropzone.addEventListener(
            "drop",
            function (event) {

                event.preventDefault();

                dropzone.classList.remove("is-dragover");


                if (event.dataTransfer) {

                    addFiles(
                        event.dataTransfer.files
                    );

                }

            }
        );

    }


    // -----------------------------
    // Upload form
    // -----------------------------

    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            function (event) {

                if (stagedFiles.length === 0) {

                    event.preventDefault();

                    return;
                }


                syncFileInput();

                uploadBtn.disabled = true;

                uploadBtn.textContent =
                    "Processing papers...";

            }
        );

    }


    // -----------------------------
    // Ask question
    // -----------------------------

    if (
        askForm &&
        questionInput &&
        askBtn
    ) {

        askForm.addEventListener(
            "submit",
            function (event) {

                const question =
                    questionInput.value.trim();


                if (!question) {

                    event.preventDefault();

                    questionInput.focus();

                    return;
                }


                askBtn.disabled = true;

                askBtn.textContent =
                    "Searching...";

            }
        );

    }


    // -----------------------------
    // Suggestion buttons
    // -----------------------------

    suggestions.forEach(
        function (suggestion) {

            suggestion.addEventListener(
                "click",
                function () {

                    const question =
                        suggestion.getAttribute(
                            "data-question"
                        );


                    if (question && questionInput) {

                        questionInput.value =
                            question;

                        questionInput.focus();

                    }

                }
            );

        }
    );

})();

