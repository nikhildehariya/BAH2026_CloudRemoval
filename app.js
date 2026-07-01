/* ==========================================================================
   Team Dhruva - Browser Application Script
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    
    // Global references to generated PNG assets
    const assets = {
        cloudy: './assets/cloudy.png',
        reconstructed: './assets/reconstructed.png',
        groundTruth: './assets/ground_truth.png',
        sarVv: './assets/sar_vv.png',
        sarComposite: './assets/sar_composite.png',
        cloudMask: './assets/cloud_mask.png'
    };

    // Preload Images
    const imgs = {};
    let imagesLoaded = 0;
    const totalImages = Object.keys(assets).length;
    
    function checkAllLoaded() {
        imagesLoaded++;
        if (imagesLoaded === totalImages) {
            console.log("All satellite assets loaded successfully.");
            initializeComponents();
        }
    }

    for (const key in assets) {
        imgs[key] = new Image();
        imgs[key].onload = checkAllLoaded;
        imgs[key].onerror = () => {
            console.warn(`Failed to load asset: ${assets[key]}. Using canvas fallbacks.`);
            checkAllLoaded();
        };
        imgs[key].src = assets[key];
    }

    // Fetch dynamic metrics from backend output
    fetch('./assets/metrics.json')
        .then(response => {
            if (!response.ok) throw new Error("Metrics file not found");
            return response.json();
        })
        .then(data => {
            console.log("Loaded dynamic metrics:", data);
            const mSam = document.getElementById("metric-sam");
            const mNdvi = document.getElementById("metric-ndvi");
            const mSsim = document.getElementById("metric-ssim");
            if (mSam) mSam.textContent = data.sam;
            if (mNdvi) mNdvi.textContent = data.ndvi;
            if (mSsim) mSsim.textContent = data.ssim;
        })
        .catch(err => {
            console.warn("Could not load dynamic metrics, using defaults:", err);
        });

    // ==========================================================================
    // Pane Navigation Switching
    // ==========================================================================
    const navItems = document.querySelectorAll(".nav-item");
    const displayPanes = document.querySelectorAll(".display-pane");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            
            // Remove active class from all navs and panes
            navItems.forEach(n => n.classList.remove("active"));
            displayPanes.forEach(p => p.classList.remove("active"));
            
            // Add active to clicked nav
            item.classList.add("active");
            
            // Reveal target pane
            const targetId = item.getAttribute("href").replace("#module-", "pane-");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
                // Trigger canvas redraws when panes become active
                triggerPaneRedraw(targetId);
            }
        });
    });

    // ==========================================================================
    // Swipe Slider Comparator
    // ==========================================================================
    const swipeBox = document.getElementById("swipe-box");
    const swipeBefore = document.getElementById("swipe-before-img");
    const swipeSlider = document.getElementById("swipe-slider-btn");

    if (swipeBox && swipeBefore && swipeSlider) {
        let isDragging = false;

        function slide(x) {
            const rect = swipeBox.getBoundingClientRect();
            let pos = (x - rect.left) / rect.width;
            
            // Restrict position to boundary
            if (pos < 0) pos = 0;
            if (pos > 1) pos = 1;
            
            swipeBefore.style.width = (pos * 100) + "%";
            swipeSlider.style.left = (pos * 100) + "%";
        }

        swipeBox.addEventListener("mousedown", (e) => {
            isDragging = true;
            slide(e.clientX);
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
        });

        window.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            slide(e.clientX);
        });

        // Touch support
        swipeBox.addEventListener("touchstart", (e) => {
            isDragging = true;
            slide(e.touches[0].clientX);
        });
        window.addEventListener("touchend", () => {
            isDragging = false;
        });
        window.addEventListener("touchmove", (e) => {
            if (!isDragging) return;
            slide(e.touches[0].clientX);
        });
    }

    // ==========================================================================
    // Module 1: Ingestion Simulator
    // ==========================================================================
    const btnRunQuery = document.getElementById("btn-run-query");
    const queryLogBox = document.getElementById("query-log-box");
    const bboxSelect = document.getElementById("bbox-selected-val");
    const thumbLiss4 = document.getElementById("thumb-liss4");
    const thumbSar = document.getElementById("thumb-sar");
    const lblLiss4 = document.getElementById("lbl-liss4");
    const lblSar = document.getElementById("lbl-sar");

    // Coordinates metadata map
    const coordsMap = {
        assam: {
            lat: "26.14° N", lon: "91.73° E",
            liss_id: "R2_L4_MX_20260615_087_054",
            sar_id: "S1A_IW_GRDH_1SDV_20260615T120000_ASC"
        },
        meghalaya: {
            lat: "25.57° N", lon: "91.88° E",
            liss_id: "R2_L4_MX_20260612_088_055",
            sar_id: "S1B_IW_GRDH_1SDV_20260612T120400_ASC"
        },
        sikkim: {
            lat: "27.33° N", lon: "88.61° E",
            liss_id: "R2_L4_MX_20260608_086_053",
            sar_id: "S1A_IW_GRDH_1SDV_20260608T115800_ASC"
        }
    };

    // Custom BBox Dropdown toggle logic
    const customDropdown = document.getElementById("bbox-dropdown");
    const dropdownItems = document.querySelectorAll(".dropdown-item");

    if (customDropdown && bboxSelect) {
        bboxSelect.addEventListener("click", (e) => {
            e.stopPropagation();
            customDropdown.classList.toggle("open");
        });

        dropdownItems.forEach(item => {
            item.addEventListener("click", (e) => {
                e.stopPropagation();
                
                // Toggle active items
                dropdownItems.forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                
                // Update selected HUD element
                const val = item.getAttribute("data-value");
                bboxSelect.textContent = item.textContent;
                bboxSelect.setAttribute("data-value", val);
                
                customDropdown.classList.remove("open");
            });
        });

        document.addEventListener("click", () => {
            customDropdown.classList.remove("open");
        });
    }

    if (btnRunQuery && queryLogBox) {
        btnRunQuery.addEventListener("click", () => {
            queryLogBox.innerHTML = "";
            const selectedLoc = bboxSelect ? bboxSelect.getAttribute("data-value") : "assam";
            const meta = coordsMap[selectedLoc];
            
            addLog("system", `[BHOONIDHI] Init query for RESOURCESAT-2 LISS-IV sensor, bbox centered at [Lat: ${meta.lat}, Lon: ${meta.lon}]...`);
            
            setTimeout(() => {
                addLog("info", `[BHOONIDHI] Scene found matching search criteria. Cloud cover: 68.5%. ID: ${meta.liss_id}`);
                addLog("system", `[CDSE] Querying Copernicus database for temporally matched Sentinel-1 GRD track...`);
            }, 800);

            setTimeout(() => {
                addLog("info", `[CDSE] Co-incident GRD track located. Match Delta: +3h 12m. ID: ${meta.sar_id}`);
                addLog("success", `[INGESTION] Download complete. Programmatic pair registered.`);
                
                // Update UI elements
                thumbLiss4.style.backgroundImage = `url(${assets.cloudy})`;
                thumbLiss4.textContent = "";
                thumbSar.style.backgroundImage = `url(${assets.sarComposite})`;
                thumbSar.textContent = "";
                
                lblLiss4.textContent = `ID: ${meta.liss_id}`;
                lblSar.textContent = `ID: ${meta.sar_id}`;
                
            }, 1800);
        });
    }

    function addLog(type, msg) {
        const entry = document.createElement("p");
        entry.className = `log-entry ${type}-msg`;
        entry.textContent = `> ${msg}`;
        queryLogBox.appendChild(entry);
        queryLogBox.scrollTop = queryLogBox.scrollHeight;
    }

    // ==========================================================================
    // Module 2: Sub-Pixel TPS Co-Registration Canvas
    // ==========================================================================
    const tpsCanvas = document.getElementById("tps-canvas");
    const rangeWarpBlend = document.getElementById("range-warp-blend");

    function drawTPSWarp() {
        if (!tpsCanvas) return;
        const ctx = tpsCanvas.getContext("2d");
        const w = tpsCanvas.width;
        const h = tpsCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const blendVal = rangeWarpBlend ? parseInt(rangeWarpBlend.value) / 100 : 0;

        // Draw Optical Background
        if (imgs.cloudy && imgs.cloudy.complete) {
            ctx.drawImage(imgs.cloudy, 0, 0, w, h);
        } else {
            ctx.fillStyle = "#1e293b";
            ctx.fillRect(0, 0, w, h);
        }

        // Draw overlay SAR with variable alpha
        if (blendVal > 0) {
            ctx.save();
            ctx.globalAlpha = blendVal;
            if (imgs.sarComposite && imgs.sarComposite.complete) {
                ctx.drawImage(imgs.sarComposite, 0, 0, w, h);
            }
            ctx.restore();
        }

        // Generate and draw tie-points (simulated local descriptors keypoints)
        const seedPoints = [
            { x: 80, y: 100, dx: -2, dy: 3 },
            { x: 200, y: 70, dx: 4, dy: -2 },
            { x: 420, y: 150, dx: -1, dy: -3 },
            { x: 130, y: 280, dx: 3, dy: 1 },
            { x: 350, y: 320, dx: -3, dy: 2 }
        ];

        seedPoints.forEach(pt => {
            // Draw Target crosshairs on optical grid
            ctx.strokeStyle = "#06b6d4";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
            ctx.moveTo(pt.x - 10, pt.y); ctx.lineTo(pt.x + 10, pt.y);
            ctx.moveTo(pt.x, pt.y - 10); ctx.lineTo(pt.x, pt.y + 10);
            ctx.stroke();

            // Draw vector displacement lines linking from unwarped SAR points
            if (blendVal < 0.8) {
                const sarX = pt.x + pt.dx;
                const sarY = pt.y + pt.dy;
                
                ctx.strokeStyle = "#10b981";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(sarX, sarY);
                ctx.lineTo(pt.x, pt.y);
                ctx.stroke();

                // Draw SAR keypoints
                ctx.fillStyle = "#10b981";
                ctx.beginPath();
                ctx.arc(sarX, sarY, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }

    if (rangeWarpBlend) {
        rangeWarpBlend.addEventListener("input", drawTPSWarp);
    }

    // ==========================================================================
    // Module 3: Two-Tier Cloud & Shadow Masking Canvas
    // ==========================================================================
    const maskCanvas = document.getElementById("mask-canvas");
    const rangeRedThresh = document.getElementById("range-red-thresh");
    const valRedThresh = document.getElementById("val-red-thresh");
    
    // Segmentation toggles
    const btnMaskRaw = document.getElementById("btn-mask-raw");
    const btnMaskSpectral = document.getElementById("btn-mask-spectral");
    const btnMaskUnet = document.getElementById("btn-mask-unet");
    
    let activeMaskMode = 'raw'; // raw, spectral, unet

    function drawMasking() {
        if (!maskCanvas) return;
        const ctx = maskCanvas.getContext("2d");
        const w = maskCanvas.width;
        const h = maskCanvas.height;
        ctx.clearRect(0, 0, w, h);

        if (activeMaskMode === 'raw') {
            if (imgs.cloudy && imgs.cloudy.complete) {
                ctx.drawImage(imgs.cloudy, 0, 0, w, h);
            }
        } 
        else if (activeMaskMode === 'unet') {
            // Draw refined Attention U-Net mask
            if (imgs.cloudMask && imgs.cloudMask.complete) {
                ctx.drawImage(imgs.cloudMask, 0, 0, w, h);
            }
        } 
        else if (activeMaskMode === 'spectral') {
            // Real-time canvas thresholding simulation
            // Draw optical first to get red band data
            if (imgs.cloudy && imgs.cloudy.complete) {
                ctx.drawImage(imgs.cloudy, 0, 0, w, h);
                const imgData = ctx.getImageData(0, 0, w, h);
                const data = imgData.data;
                const thresh = rangeRedThresh ? parseInt(rangeRedThresh.value) / 100 * 255 : 64;
                
                for (let i = 0; i < data.length; i += 4) {
                    // Extract Red band channel
                    const redVal = data[i]; 
                    
                    // Simple thresholding
                    const binaryVal = (redVal > thresh) ? 255 : 0;
                    
                    data[i] = binaryVal;
                    data[i+1] = binaryVal;
                    data[i+2] = binaryVal;
                }
                ctx.putImageData(imgData, 0, 0);
            }
        }
    }

    if (rangeRedThresh && valRedThresh) {
        rangeRedThresh.addEventListener("input", () => {
            const val = parseFloat(rangeRedThresh.value) / 100;
            valRedThresh.textContent = val.toFixed(2);
            if (activeMaskMode === 'spectral') {
                drawMasking();
            }
        });
    }

    const setMaskMode = (mode, activeBtn) => {
        activeMaskMode = mode;
        [btnMaskRaw, btnMaskSpectral, btnMaskUnet].forEach(btn => btn.classList.remove("active"));
        activeBtn.classList.add("active");
        drawMasking();
    };

    if (btnMaskRaw) btnMaskRaw.addEventListener("click", () => setMaskMode('raw', btnMaskRaw));
    if (btnMaskSpectral) btnMaskSpectral.addEventListener("click", () => setMaskMode('spectral', btnMaskSpectral));
    if (btnMaskUnet) btnMaskUnet.addEventListener("click", () => setMaskMode('unet', btnMaskUnet));


    // ==========================================================================
    // Module 4: Latent Diffusion Denoising Simulator
    // ==========================================================================
    const diffusionCanvas = document.getElementById("diffusion-canvas");
    const btnDiffPlay = document.getElementById("btn-diff-play");
    const rangeDiffStep = document.getElementById("range-diff-step");
    const lblDiffStep = document.getElementById("lbl-diff-step");
    const btnDiffViewRecon = document.getElementById("btn-diff-view-recon");
    const btnDiffViewAttn = document.getElementById("btn-diff-view-attn");

    let isDenoisingPlaying = false;
    let diffusionPlayInterval = null;
    let activeDiffView = 'recon'; // recon, attn

    function drawDiffusionStep() {
        if (!diffusionCanvas) return;
        const ctx = diffusionCanvas.getContext("2d");
        const w = diffusionCanvas.width;
        const h = diffusionCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const step = rangeDiffStep ? parseInt(rangeDiffStep.value) : 0;
        lblDiffStep.textContent = `Step ${step}/10`;

        if (activeDiffView === 'recon') {
            // Reconstruct timeline from random Gaussian noise (step 0) to clean target (step 10)
            if (step === 0) {
                // Generate raw random noise grid
                const imgData = ctx.createImageData(w, h);
                const data = imgData.data;
                for (let i = 0; i < data.length; i += 4) {
                    const noise = Math.floor(Math.random() * 256);
                    data[i] = noise;
                    data[i+1] = noise;
                    data[i+2] = noise;
                    data[i+3] = 255;
                }
                ctx.putImageData(imgData, 0, 0);
            } 
            else {
                // Blend noise, raw cloudy image and clean reconstruction based on step number
                // Under cloudy mask pixels we resolve to clean, under clear pixels we resolve to original optical
                ctx.drawImage(imgs.cloudy, 0, 0, w, h);
                const optData = ctx.getImageData(0, 0, w, h).data;
                
                ctx.drawImage(imgs.reconstructed, 0, 0, w, h);
                const reconData = ctx.getImageData(0, 0, w, h).data;

                ctx.drawImage(imgs.cloudMask, 0, 0, w, h);
                const maskData = ctx.getImageData(0, 0, w, h).data;

                const finalImgData = ctx.createImageData(w, h);
                const finalData = finalImgData.data;

                const blendFactor = step / 10.0; // resolve progression

                for (let i = 0; i < finalData.length; i += 4) {
                    const isCloud = maskData[i] > 128; // mask is active
                    
                    if (isCloud) {
                        // resolve from noise to reconstruction
                        const noiseVal = Math.floor(Math.random() * 256);
                        finalData[i] = noiseVal * (1.0 - blendFactor) + reconData[i] * blendFactor;
                        finalData[i+1] = noiseVal * (1.0 - blendFactor) + reconData[i+1] * blendFactor;
                        finalData[i+2] = noiseVal * (1.0 - blendFactor) + reconData[i+2] * blendFactor;
                    } else {
                        // Clear-sky pixels are preserved pristine from the input optical bands
                        finalData[i] = optData[i];
                        finalData[i+1] = optData[i+1];
                        finalData[i+2] = optData[i+2];
                    }
                    finalData[i+3] = 255;
                }
                ctx.putImageData(finalImgData, 0, 0);
            }
        } 
        else if (activeDiffView === 'attn') {
            // Draw cross-attention weight maps
            // Glows hot violet/red on mask boundary, fading into low-attention blue on clear sky
            if (imgs.cloudMask && imgs.cloudMask.complete) {
                ctx.drawImage(imgs.cloudMask, 0, 0, w, h);
                const maskData = ctx.getImageData(0, 0, w, h).data;
                const finalImgData = ctx.createImageData(w, h);
                const finalData = finalImgData.data;

                for (let i = 0; i < finalData.length; i += 4) {
                    const maskVal = maskData[i]; // 0 or 255
                    
                    if (maskVal > 128) {
                        // High Attention region: map to hot neon red/pink
                        // indicating active routing to S-1 SAR keys
                        finalData[i] = 239;
                        finalData[i+1] = 68;
                        finalData[i+2] = 68;
                    } else {
                        // Self-attention optical region: violet/blue tones
                        finalData[i] = 139 - maskVal;
                        finalData[i+1] = 92;
                        finalData[i+2] = 246;
                    }
                    finalData[i+3] = 200; // opacity
                }
                ctx.putImageData(finalImgData, 0, 0);
            }
        }
    }

    if (rangeDiffStep) {
        rangeDiffStep.addEventListener("input", drawDiffusionStep);
    }

    if (btnDiffPlay) {
        btnDiffPlay.addEventListener("click", () => {
            if (isDenoisingPlaying) {
                // Pause
                clearInterval(diffusionPlayInterval);
                btnDiffPlay.textContent = "▶ Run Diffusion";
                isDenoisingPlaying = false;
            } else {
                // Start Playback loop
                btnDiffPlay.textContent = "⏸ Pause";
                isDenoisingPlaying = true;
                
                if (parseInt(rangeDiffStep.value) >= 10) {
                    rangeDiffStep.value = 0;
                }
                
                diffusionPlayInterval = setInterval(() => {
                    let currentVal = parseInt(rangeDiffStep.value);
                    currentVal++;
                    rangeDiffStep.value = currentVal;
                    drawDiffusionStep();
                    
                    if (currentVal >= 10) {
                        clearInterval(diffusionPlayInterval);
                        btnDiffPlay.textContent = "▶ Run Diffusion";
                        isDenoisingPlaying = false;
                    }
                }, 300);
            }
        });
    }

    const setDiffViewMode = (mode, activeBtn) => {
        activeDiffView = mode;
        [btnDiffViewRecon, btnDiffViewAttn].forEach(btn => btn.classList.remove("active"));
        activeBtn.classList.add("active");
        drawDiffusionStep();
    };

    if (btnDiffViewRecon) btnDiffViewRecon.addEventListener("click", () => setDiffViewMode('recon', btnDiffViewRecon));
    if (btnDiffViewAttn) btnDiffViewAttn.addEventListener("click", () => setDiffViewMode('attn', btnDiffViewAttn));


    // ==========================================================================
    // Module 5: Seamless Post-Processing & Stitcher Canvas
    // ==========================================================================
    const stitchCanvas = document.getElementById("stitch-canvas");
    const blendBlocky = document.getElementById("blend-blocky");
    const blendGaussian = document.getElementById("blend-gaussian");
    const rangeGaussianSigma = document.getElementById("range-gaussian-sigma");
    const valGaussianSigma = document.getElementById("val-gaussian-sigma");

    function drawStitching() {
        if (!stitchCanvas) return;
        const ctx = stitchCanvas.getContext("2d");
        const w = stitchCanvas.width;
        const h = stitchCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const isSeamless = blendGaussian && blendGaussian.checked;
        
        if (imgs.reconstructed && imgs.reconstructed.complete) {
            ctx.drawImage(imgs.reconstructed, 0, 0, w, h);
            
            if (!isSeamless) {
                // Draw harsh block grid overlap seam lines to visualize competitor stitching artifacts
                ctx.strokeStyle = "rgba(255, 0, 0, 0.4)";
                ctx.lineWidth = 2.0;
                
                const patchGrid = [128, 256, 384];
                patchGrid.forEach(line => {
                    // Vertical borders
                    ctx.beginPath();
                    ctx.moveTo(line, 0); ctx.lineTo(line, h);
                    ctx.stroke();
                    // Horizontal borders
                    ctx.beginPath();
                    ctx.moveTo(0, line); ctx.lineTo(w, line);
                    ctx.stroke();
                });
                
                // Text label warning
                ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
                ctx.font = "bold 11px sans-serif";
                ctx.fillText("SEAM ALIGNMENT LINES ACTIVE", 14, 24);
            } else {
                ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
                ctx.font = "bold 11px sans-serif";
                ctx.fillText("GAUSSIAN BOUNDARY BLENDING ACTIVE", 14, 24);
            }
        }
    }

    if (blendBlocky) blendBlocky.addEventListener("change", drawStitching);
    if (blendGaussian) blendGaussian.addEventListener("change", drawStitching);
    if (rangeGaussianSigma && valGaussianSigma) {
        rangeGaussianSigma.addEventListener("input", () => {
            valGaussianSigma.textContent = rangeGaussianSigma.value + " px";
        });
    }

    // Programmatic simulation of GeoTIFF download click
    const btnDownloadCog = document.getElementById("btn-download-cog");
    if (btnDownloadCog) {
        btnDownloadCog.addEventListener("click", () => {
            alert("Initiating georeferenced Cloud-Optimized GeoTIFF (COG) export...\nFile saved to: ./data/processed/TeamDhruva_LISS4_CloudFree.tif");
        });
    }


    // ==========================================================================
    // Module 6: Judges Defense Q&A Portal Toggle
    // ==========================================================================
    const qCards = document.querySelectorAll(".defense-q-card");
    qCards.forEach(card => {
        card.addEventListener("click", () => {
            const wasExpanded = card.classList.contains("expanded");
            
            // Collapse all
            qCards.forEach(c => c.classList.remove("expanded"));
            
            // Expand current if it wasn't expanded
            if (!wasExpanded) {
                card.classList.add("expanded");
            }
        });
    });

    // ==========================================================================
    // Live Terminal Logger Simulation
    // ==========================================================================
    const termLog = document.getElementById("live-terminal-log");
    const terminalLogsList = [
        "[OIDC AUTH] Authenticated with Copernicus CDSE OIDC successfully.",
        "[PREPROCESSING] Extracting metadata: SOLAR_ZENITH=32.5, SOLAR_AZIMUTH=122.4, DOY=166",
        "[PREPROCESSING] Running Py6S Top-Of-Atmosphere physical correction baseline...",
        "[PREPROCESSING] Computing astronomical distance correction factor: d = 0.9856",
        "[PREPROCESSING] Applying local-statistics Refined Lee Filter to Sentinel-1 channels...",
        "[CO-REGISTRATION] Extracting SIFT tie-points between LISS-IV and Sentinel-1...",
        "[CO-REGISTRATION] SIFT feature matches insufficient. Fallback to robust cross-correlation.",
        "[CO-REGISTRATION] Solved Thin-Plate Spline coefficients. RMSE = 0.12 pixels.",
        "[MASKING] Generating Tier-1 spectral Red-band threshold cloud guess (thresh=0.25)...",
        "[MASKING] Initializing Tier-2 Attention U-Net segmenter on active device (CPU)...",
        "[MASKING] Epoch 1/1 complete. Binary Cross Entropy Loss: 0.75877.",
        "[DIFFUSION] Encoding optical patches via KL-Autoencoder space-compression...",
        "[DIFFUSION] Initializing reverse latent diffusion denoising loop (T=10 steps)...",
        "[DIFFUSION] Decoupling cross-attention: routing to Sentinel-1 SAR structures (M=1)...",
        "[POSTPROCESSING] Executing row-by-row patch accumulation on 256x256 grids...",
        "[POSTPROCESSING] Invoking gc.collect() and torch.cuda.empty_cache() to free VRAM...",
        "[POSTPROCESSING] Seamless 2D Gaussian stitching boundary blend complete (sigma=25px).",
        "[EXPORT] Writing georeferenced metadata UTM zone 46N to TeamDhruva_LISS4_CloudFree.tif",
        "[EVALUATION] Mean Absolute Error (L1): 0.33820, Multi-Scale SSIM: 0.60319",
        "[EVALUATION] SAM: 0.58068 rad, NDVI Consistency: 42.05%, Joint Loss: 0.82660"
    ];

    let logIndex = 0;
    function streamTerminalLogs() {
        if (!termLog) return;
        
        const entry = document.createElement("p");
        entry.className = "term-entry";
        entry.innerHTML = `<span class="term-time">[${new Date().toLocaleTimeString()}]</span> ${terminalLogsList[logIndex % terminalLogsList.length]}`;
        termLog.appendChild(entry);
        termLog.scrollTop = termLog.scrollHeight;
        
        logIndex++;
        setTimeout(streamTerminalLogs, 2500 + Math.random() * 2500);
    }

    // ==========================================================================
    // Initializer and Redraw Routing
    // ==========================================================================
    function initializeComponents() {
        drawTPSWarp();
        drawMasking();
        drawDiffusionStep();
        drawStitching();
        streamTerminalLogs();
    }

    function triggerPaneRedraw(paneId) {
        // Redraw canvas elements when panel shifts to avoid canvas aspect issues
        setTimeout(() => {
            if (paneId === "pane-preprocessing") drawTPSWarp();
            if (paneId === "pane-masking") drawMasking();
            if (paneId === "pane-diffusion") drawDiffusionStep();
            if (paneId === "pane-postprocessing") drawStitching();
        }, 50);
    }
});
