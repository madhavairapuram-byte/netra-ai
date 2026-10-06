const API_URL = "http://127.0.0.1:8000/predict";


// ========================================================
// SCREEN ELEMENTS
// ========================================================

const homeScreen = document.getElementById("home-screen");
const captureScreen = document.getElementById("capture-screen");
const resultScreen = document.getElementById("result-screen");

const startButton = document.getElementById("start-button");
const fileInput = document.getElementById("file-input");

const newScreeningButton =
    document.getElementById("new-screening") ||
    document.getElementById("new-screening-button");

const statusText = document.getElementById("status");

const resultCard = document.getElementById("result-card");
const resultIcon = document.getElementById("result-icon");
const resultTitle = document.getElementById("result-title");
const resultDetail = document.getElementById("result-detail");
const confidenceText = document.getElementById("confidence");

const diseaseInfo = document.getElementById("disease-info");

const diseaseDescription =
    document.getElementById("disease-description");

const diseaseAction =
    document.getElementById("disease-action");

const diseaseTiming =
    document.getElementById("disease-timing");


// ========================================================
// DISEASE INFORMATION
// ========================================================

const diseaseInformation = {

    Cataract: {

        description:
            "A cataract is a clouding of the eye's natural lens. " +
            "It commonly develops gradually and can cause blurred, " +
            "hazy, or less clear vision.",

        action:
            "Arrange an eye examination with an eye-care professional. " +
            "Until then, avoid driving or activities where reduced vision " +
            "could make you unsafe. Wearing sunglasses that provide UV " +
            "protection can also help protect your eyes.",

        timing:
            "Cataracts are usually gradual rather than sudden. Seek urgent " +
            "medical attention if you develop sudden vision loss, severe eye " +
            "pain, or a major sudden change in vision."
    },


    conjunctivitis: {

        description:
            "Conjunctivitis is inflammation of the conjunctiva, the thin " +
            "surface covering the white part of the eye and the inside of " +
            "the eyelids. It can cause redness, irritation, watering, or discharge.",

        action:
            "Avoid rubbing your eyes. Wash your hands regularly and do not " +
            "share towels, cosmetics, or eye products. Avoid contact lenses " +
            "while the eye is affected and seek advice from an eye-care professional.",

        timing:
            "Seek medical attention promptly if there is severe eye pain, " +
            "strong sensitivity to light, significant swelling, vision changes, " +
            "or symptoms that are getting worse."
    },


    stye: {

        description:
            "A stye is a localized, usually painful swelling of an eyelid, " +
            "often associated with inflammation around an eyelash follicle or " +
            "an eyelid gland.",

        action:
            "Apply a clean warm compress to the closed eyelid for several " +
            "minutes at a time. Do not squeeze or try to burst the stye. " +
            "Avoid eye makeup and contact lenses while the area is affected.",

        timing:
            "Seek medical attention if swelling or redness spreads, you develop " +
            "fever, your vision changes, the pain becomes significant, or the " +
            "problem is not improving."
    }
};


// ========================================================
// SCREEN MANAGEMENT
// ========================================================

function showScreen(screen) {

    if (!screen) {
        return;
    }

    if (homeScreen) {
        homeScreen.classList.remove("active");
    }

    if (captureScreen) {
        captureScreen.classList.remove("active");
    }

    if (resultScreen) {
        resultScreen.classList.remove("active");
    }

    screen.classList.add("active");
}


// ========================================================
// START SCREENING
// ========================================================

if (startButton) {

    startButton.addEventListener("click", () => {

        if (fileInput) {
            fileInput.value = "";
        }

        if (statusText) {
            statusText.textContent = "";
        }

        showScreen(captureScreen);
    });
}


// ========================================================
// FILE SELECTION
// ========================================================

if (fileInput) {

    fileInput.addEventListener("change", async () => {

        const file = fileInput.files[0];

        if (!file) {
            return;
        }

        await runPrediction(file);
    });
}


// ========================================================
// RUN PREDICTION
// ========================================================

async function runPrediction(file) {

    if (statusText) {
        statusText.textContent =
            "Analysing your eye image...";
    }

    const formData = new FormData();

    formData.append("file", file);

    try {

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });


        if (!response.ok) {

            let errorMessage =
                `Server returned ${response.status}.`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {
                    errorMessage =
                        errorData.detail;
                }

            } catch (_) {
                // Ignore JSON parsing errors.
            }

            throw new Error(errorMessage);
        }


        const result =
            await response.json();

        displayResult(result);

    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        if (statusText) {
            statusText.textContent =
                "Prediction failed. Please make sure the NETRA AI backend is running.";
        }

        alert(
            "Prediction failed.\n\n" +
            error.message +
            "\n\n" +
            "Please make sure the backend is running on port 8000."
        );
    }
}


// ========================================================
// DISPLAY RESULT
// ========================================================

function displayResult(result) {

    const label = result.label;

    const probability =
        Number(result.probability || 0);

    const confidence =
        probability * 100;


    if (!resultCard) {
        return;
    }


    // Reset previous result styling.

    resultCard.classList.remove(
        "result-low",
        "result-medium",
        "result-high"
    );


    if (diseaseInfo) {
        diseaseInfo.classList.add("hidden");
    }


    if (resultIcon) {
        resultIcon.textContent = "";
    }


    if (resultTitle) {
        resultTitle.textContent = "";
    }


    if (resultDetail) {
        resultDetail.textContent = "";
    }


    if (confidenceText) {
        confidenceText.textContent = "";
    }


    // ====================================================
    // INVALID IMAGE
    // ====================================================

    if (label === "invalid_image") {

        resultCard.classList.add(
            "result-medium"
        );

        if (resultIcon) {
            resultIcon.textContent = "!";
        }

        if (resultTitle) {
            resultTitle.textContent =
                "Please upload an eye image.";
        }

        if (resultDetail) {
            resultDetail.textContent =
                "The image did not appear to contain a human eye. " +
                "Please upload a clear photograph of an eye.";
        }

        showScreen(resultScreen);

        return;
    }


    // ====================================================
    // NORMAL
    // ====================================================

    if (label === "normal") {

        resultCard.classList.add(
            "result-low"
        );

        if (resultIcon) {
            resultIcon.textContent = "✓";
        }

        if (resultTitle) {
            resultTitle.textContent =
                "No abnormality identified";
        }

        if (resultDetail) {
            resultDetail.textContent =
                "The AI model did not identify one of the four abnormalities " +
                "included in this screening prototype.";
        }

        showScreen(resultScreen);

        return;
    }


    // ====================================================
    // DISEASE NAME
    // ====================================================

    const disease =
        diseaseInformation[label];

    let displayName =
        label;


    if (label === "conjunctivitis") {

        displayName =
            "Conjunctivitis";

    } else if (label === "Cataract") {

        displayName =
            "Cataract";

    } else if (label === "stye") {

        displayName =
            "Stye";
    }


    // ====================================================
    // STYE
    //
    // 0–90%  = Normal
    // 91–95% = High probability
    // 96–100% = Stye detected
    // ====================================================

    if (label === "stye") {

        if (confidence <= 90) {

            showNormalDiseaseResult(
                displayName
            );

            return;
        }


        if (confidence <= 95) {

            showHighProbabilityResult(
                displayName,
                disease
            );

            return;
        }


        showDetectedResult(
            displayName,
            disease
        );

        return;
    }


    // ====================================================
    // CATARACT
    //
    // 0–90%  = Normal
    // 91–95% = High probability
    // 96–100% = Cataract detected
    // ====================================================

    if (label === "Cataract") {

        if (confidence <= 90) {

            showNormalDiseaseResult(
                displayName
            );

            return;
        }


        if (confidence <= 95) {

            showHighProbabilityResult(
                displayName,
                disease
            );

            return;
        }


        showDetectedResult(
            displayName,
            disease
        );

        return;
    }


    // ====================================================
    // CONJUNCTIVITIS
    //
    // 0–85%  = Normal
    // 86–90% = High probability
    // 91–100% = Conjunctivitis detected
    // ====================================================

    if (label === "conjunctivitis") {

        if (confidence <= 85) {

            showNormalDiseaseResult(
                displayName
            );

            return;
        }


        if (confidence <= 90) {

            showHighProbabilityResult(
                displayName,
                disease
            );

            return;
        }


        showDetectedResult(
            displayName,
            disease
        );

        return;
    }


    // ====================================================
    // UNKNOWN CLASS
    // ====================================================

    resultCard.classList.add(
        "result-medium"
    );

    if (resultIcon) {
        resultIcon.textContent = "!";
    }

    if (resultTitle) {
        resultTitle.textContent =
            "Unable to classify image.";
    }

    if (resultDetail) {
        resultDetail.textContent =
            "The model returned an unexpected result. " +
            "Please try another clear eye photograph.";
    }

    showScreen(resultScreen);
}


// ========================================================
// NORMAL RESULT FOR A DISEASE CLASS
// ========================================================

function showNormalDiseaseResult(displayName) {

    resultCard.classList.add(
        "result-low"
    );


    if (resultIcon) {
        resultIcon.textContent =
            "✓";
    }


    if (resultTitle) {
        resultTitle.textContent =
            "No abnormality identified";
    }


    if (resultDetail) {
        resultDetail.textContent =
            `The screening did not identify ${displayName.toLowerCase()} ` +
            "at a level requiring a flagged result.";
    }


    showScreen(resultScreen);
}


// ========================================================
// HIGH PROBABILITY RESULT
// ========================================================

function showHighProbabilityResult(
    displayName,
    disease
) {

    resultCard.classList.add(
        "result-medium"
    );


    if (resultIcon) {
        resultIcon.textContent =
            "!";
    }


    if (resultTitle) {
        resultTitle.textContent =
            `High probability of ${displayName}`;
    }


    if (resultDetail) {
        resultDetail.textContent =
            "The AI model identified features associated with this condition. " +
            "This is a screening result and should not be considered a diagnosis. " +
            "Consider confirmation by a qualified healthcare professional.";
    }


    showDiseaseInformation(
        disease
    );


    showScreen(resultScreen);
}


// ========================================================
// DETECTED RESULT
// ========================================================

function showDetectedResult(
    displayName,
    disease
) {

    resultCard.classList.add(
        "result-high"
    );


    if (resultIcon) {
        resultIcon.textContent =
            "!";
    }


    if (resultTitle) {
        resultTitle.textContent =
            `${displayName} detected`;
    }


    if (resultDetail) {
        resultDetail.textContent =
            "The AI model identified features strongly associated with this condition. " +
            "This is a screening result and should not be considered a diagnosis. " +
            "Please arrange confirmation with a qualified healthcare professional.";
    }


    showDiseaseInformation(
        disease
    );


    showScreen(resultScreen);
}


// ========================================================
// DISEASE INFORMATION DISPLAY
// ========================================================

function showDiseaseInformation(disease) {

    if (!diseaseInfo || !disease) {
        return;
    }


    if (diseaseDescription) {

        diseaseDescription.textContent =
            disease.description;
    }


    if (diseaseAction) {

        diseaseAction.textContent =
            disease.action;
    }


    if (diseaseTiming) {

        diseaseTiming.textContent =
            disease.timing;
    }


    diseaseInfo.classList.remove(
        "hidden"
    );
}


// ========================================================
// NEW SCREENING
// ========================================================

if (newScreeningButton) {

    newScreeningButton.addEventListener(
        "click",
        () => {

            if (fileInput) {
                fileInput.value = "";
            }

            if (statusText) {
                statusText.textContent = "";
            }

            if (diseaseInfo) {
                diseaseInfo.classList.add(
                    "hidden"
                );
            }

            showScreen(homeScreen);
        }
    );
}