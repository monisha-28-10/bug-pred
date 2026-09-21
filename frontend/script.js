
async function analyzeGithub() {

    const input = document.getElementById("githubUrl");
    const status = document.getElementById("githubStatus");

    const url = input.value.trim();

    console.log("GitHub URL entered:", url);

    const button = document.getElementById("githubBtn");

    if (!url) {
        status.innerText = "Please enter a GitHub repository URL.";
        status.className = "error";
        return;
    }

    // Accept GitHub repository URLs
    const githubPattern =
        /^https:\/\/github\.com\/[^\/\s]+\/[^\/\s]+\/?$/;

    if (!githubPattern.test(url)) {
        status.innerText =
            "Please enter a valid GitHub repository URL.";
        status.className = "error";
        return;
    }

    status.innerText = "Analyzing repository... Please wait.";
    status.className = "analyzing";
    button.innerText = "Analyzing...";
    button.disabled = true;

    try {

        const response = await fetch(
            "https://bug-pred-api.onrender.com/analyze-github",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    github_url: url
                })
            }
        );

        const result = await response.json();

        console.log("Backend response:", result);

        if (!result.success) {
            status.innerText = result.message;
            status.className = "error";
            button.innerText = "Analyze Repository";
            button.disabled = false;
            return;
        }

        status.innerText =
            `Analysis completed successfully. Found ${result.source_file_count} source files.`;

        status.className = "success";

        const githubResults = document.getElementById("githubResults");

        githubResults.innerHTML = "";

        if (result.github_predictions && result.github_predictions.length > 0) {

            const validPredictions = result.github_predictions.filter(
                item => !item.error
            );

            if (validPredictions.length > 0) {

                const highestRiskFile = validPredictions.reduce(
                    (highest, current) =>
                        current.bug_probability > highest.bug_probability
                            ? current
                            : highest
                );
                

                const summary = document.createElement("div");
                summary.className = "github-summary";

                summary.innerHTML = `
                    <h3>Repository Analysis Summary</h3>

                    <div class="github-summary-grid">

                        <div>
                            <span>Source Files</span>
                            <strong>${validPredictions.length}</strong>
                        </div>

                        <div>
                            <span>Highest Risk File</span>
                            <strong>${highestRiskFile.file}</strong>
                        </div>

                        <div>
                            <span>Highest Risk</span>
                            <strong class="risk-${highestRiskFile.risk_level.toLowerCase()}">
                                ${highestRiskFile.risk_level}
                            </strong>
                        </div>

                        <div>
                            <span>Highest Probability</span>
                            <strong>${highestRiskFile.bug_probability}%</strong>
                        </div>

                    </div>
                `;

                githubResults.appendChild(summary);
            }
        }

        if (result.github_predictions && result.github_predictions.length > 0) {

            const heading = document.createElement("h3");
            heading.innerText = "Bug Risk by Source File";

            githubResults.appendChild(heading);

            result.github_predictions.forEach(item => {

                const card = document.createElement("div");

                const highestRisk = Math.max(
                    ...result.github_predictions
                        .filter(file => !file.error)
                        .map(file => file.bug_probability)
                );

                if (!item.error && item.bug_probability === highestRisk) {
                    card.className = "github-risk-card highest-risk";
                } else {
                    card.className = "github-risk-card";
                }

                if (item.error) {

                    card.innerHTML = `
                        <div class="github-file-name">
                            ${item.file}
                        </div>

                        <p>Error: ${item.error}</p>
                    `;

                } else {

                    let factorsHTML = "";

                    if (item.risk_factors && item.risk_factors.length > 0) {

                        const topFactors = item.risk_factors.slice(0, 3);
                        const remainingFactors = item.risk_factors.slice(3);

                        // Show only top 3
                        topFactors.forEach(factor => {

                            factorsHTML += `
                                <div class="github-factor">

                                    <div class="github-factor-header">
                                        <strong>${factor.feature}</strong>

                                        <span>
                                            ${factor.risk_contribution}%
                                        </span>
                                    </div>

                                    <p>
                                        <strong>Value:</strong>
                                        ${factor.value}
                                    </p>

                                    <p>
                                        <strong>Explanation:</strong>
                                        ${factor.explanation}
                                    </p>

                                    <p>
                                        <strong>Recommendation:</strong>
                                        ${factor.recommendation}
                                    </p>

                                </div>
                            `;
                        });

                        // Show remaining factors in compact form
                        if (remainingFactors.length > 0) {

                            const remainingText = remainingFactors
                                .map(factor =>
                                    `${factor.feature} (${factor.risk_contribution}%)`
                                )
                                .join(", ");

                            factorsHTML += `
                                <div class="github-remaining">
                                    <strong>Remaining:</strong> ${remainingText}
                                </div>
                            `;
                        }
                    }

                    card.innerHTML = `

                        <div class="github-file-header">

                            <div class="github-file-name">
                                ${item.file}
                            </div>
                            ${
                                item.bug_probability === highestRisk
                                    ? `<span class="highest-risk-badge">Highest Risk</span>`
                                    : ""
                            }

                        </div>

                        <div class="github-risk-info">

                            <span>
                                Bug Probability:
                                <strong>${item.bug_probability}%</strong>
                            </span>

                            <span>
                                Risk:
                                <strong class="risk-${item.risk_level.toLowerCase()}">
                                    ${item.risk_level}
                                </strong>
                            </span>

                        </div>

                        ${
                        item.risk_level === "HIGH" ||
                        item.risk_level === "CRITICAL"
                            ? `
                                <div class="github-details">
                                    <h4>Top Risk Factors</h4>

                                    ${factorsHTML}

                                    <div class="github-overall-recommendation">
                                        <h4>Overall Recommendation</h4>
                                        <p>
                                            ${item.overall_recommendation}
                                        </p>
                                    </div>

                                </div>
                            `
                            : ""
                        }
                    `;
                }

                githubResults.appendChild(card);
            });
        }
        button.innerText = "Analyze Repository";
        button.disabled = false;

        console.log("Source code:", result.source_contents);

    } catch (error) {

        console.error("GitHub analysis error:", error);

        status.innerText =
            "Unable to connect to the prediction server.";
        status.className = "error";
        button.innerText = "Analyze Repository";
        button.disabled = false;
    }
}