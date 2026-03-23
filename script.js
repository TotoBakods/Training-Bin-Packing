// Global variables
let scene, camera, renderer, controls;
let warehouseGroup, itemsGroup, exclusionZonesGroup, pickerPathGroup;
let coordinateLabels = [];
let warehouseConfig = {};
let allItemsData = {};

let isOptimizing = false;
let optimizationInterval;
let historyChart, categoryChart;
let currentWarehouseId = 1;
let warehouses = [];

// API Base URL
const API_BASE_URL = window.location.origin; // Use current origin

document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    loadWarehouses();
    loadWarehouseConfig(); // Initial load
    loadZones(); // Ensure zones are loaded on startup
    switchView('controls');
    updateAlgorithmParams(); // Initialize algorithm params
});

// Algorithm parameter UI
function updateAlgorithmParams() {
    // The ML model no longer uses configurable parameters in the UI.
    const container = document.getElementById('algorithm-params');
    const compareParams = document.getElementById('compare-params');
    
    if (compareParams) compareParams.style.display = 'none';
    if (container) container.style.display = 'none';
}

// UI navigation
function switchView(viewName) {
    // Hide all view panels
    document.querySelectorAll('.view-panel').forEach(el => el.style.display = 'none');
    // Show selected
    document.getElementById(`view-${viewName}`).style.display = 'block';

    // Update active nav state
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    // Simple mapping based on click handling, or add IDs to nav-items
    // For now assuming the click handler works, but let's visually update if possible
    // (This part relies on the onclick adding 'active' class, which we can do manually here if we had IDs)

    if (viewName === 'analytics') loadAnalytics();
    if (viewName === 'items') loadItemsList();
    if (viewName === 'warehouse') loadWarehouseConfig();
}

// Three.js visualization
function initThreeJS() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505); // Match CSS --bg-main

    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(20, 20, 20);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(10, 30, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    // Groups
    warehouseGroup = new THREE.Group();
    itemsGroup = new THREE.Group();
    exclusionZonesGroup = new THREE.Group();
    pickerPathGroup = new THREE.Group();
    scene.add(warehouseGroup);
    scene.add(itemsGroup);
    scene.add(exclusionZonesGroup);
    scene.add(pickerPathGroup);

    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('click', onMouseClick);

    animate();
}

function onWindowResize() {
    const container = document.getElementById('three-container');
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Raycasting for tooltips
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseClick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Check if place door mode is active
    const placeDoorToggle = document.getElementById('place-door-btn');
    const isEditingDoor = placeDoorToggle && placeDoorToggle.classList.contains('active-tool');
    
    if (event.shiftKey || isEditingDoor) {
        // Intersect with floor
        const floorIntersects = raycaster.intersectObjects(warehouseGroup.children);
        const floorHit = floorIntersects.find(hit => hit.object.name === "warehouse_floor");
        
        if (floorHit) {
            // Hit point is relative to warehouse center, convert to normal coordinates (0 to length/width)
            const newX = floorHit.point.x + warehouseConfig.length / 2;
            const newY = floorHit.point.z + warehouseConfig.width / 2;
            
            // Snap to nearest wall to make it realistic
            const dl = newX;
            const dr = warehouseConfig.length - newX;
            const dt = newY;
            const db = warehouseConfig.width - newY;
            const minDist = Math.min(dl, dr, dt, db);
            
            let snappedX = newX;
            let snappedY = newY;
            
            if (minDist === dl) snappedX = 0;
            else if (minDist === dr) snappedX = warehouseConfig.length;
            else if (minDist === dt) snappedY = 0;
            else if (minDist === db) snappedY = warehouseConfig.width;
            
            // Limit to bounds
            snappedX = Math.max(0, Math.min(warehouseConfig.length, snappedX));
            snappedY = Math.max(0, Math.min(warehouseConfig.width, snappedY));
            
            // Update UI
            document.getElementById('warehouse-door-x').value = snappedX.toFixed(1);
            document.getElementById('warehouse-door-y').value = snappedY.toFixed(1);

            // Show confirmation button
            const confirmBtn = document.getElementById('confirm-door-btn');
            if (confirmBtn) {
                confirmBtn.style.display = 'block';
                confirmBtn.innerText = `CONFIRM DOOR AT (${snappedX.toFixed(1)}, ${snappedY.toFixed(1)})`;
            }
            return;
        }
    }

    const intersects = raycaster.intersectObjects(itemsGroup.children);

    const tooltip = document.getElementById('tooltip');

    if (intersects.length > 0) {
        const data = intersects[0].object.userData;
        tooltip.style.display = 'block';
        tooltip.style.left = (event.clientX + 15) + 'px';
        tooltip.style.top = (event.clientY + 15) + 'px';
        tooltip.innerHTML = `
            <strong>${data.name || 'Item ' + data.id}</strong>
            <div style="font-size:0.8rem; color:#bbb;">
                ${data.width}x${data.height}x${data.length}m<br>
                Weight: ${data.weight}kg<br>
                Pos: ${data.x.toFixed(2)}, ${data.y.toFixed(2)}, ${data.z.toFixed(2)}<br>
                Fragile: ${data.fragility ? '<span style="color:#FF6B8A">Yes ⚠️</span>' : '<span style="color:#00FF9D">No</span>'}
            </div>
        `;
        // Click selection could go here
    } else {
        tooltip.style.display = 'none';
    }
}

function togglePlaceDoorMode() {
    const btn = document.getElementById('place-door-btn');
    const confirmBtn = document.getElementById('confirm-door-btn');
    if (!btn) return;

    if (btn.classList.contains('active-tool')) {
        btn.classList.remove('active-tool');
        btn.style.background = '';
        btn.style.color = '';
        btn.innerText = 'PLACE DOOR (Click on floor)';
        if (confirmBtn) confirmBtn.style.display = 'none';
    } else {
        btn.classList.add('active-tool');
        btn.style.background = 'var(--accent-primary)';
        btn.style.color = '#000';
        btn.innerText = 'CANCEL PLACEMENT';
    }
}

function confirmDoorPlacement() {
    updateWarehouseConfig();
    togglePlaceDoorMode();
}

// Data loading and rendering

function loadWarehouses() {
    fetch(`${API_BASE_URL}/api/warehouses`)
        .then(res => res.json())
        .then(data => {
            warehouses = data;
            const select = document.getElementById('warehouse-select');
            select.innerHTML = '';
            data.forEach(w => {
                const opt = document.createElement('option');
                opt.value = w.id;
                opt.textContent = w.name;
                select.appendChild(opt);
            });
            if (data.length > 0) {
                currentWarehouseId = data[0].id;
                loadWarehouseConfig();
            }
        });
}

function switchWarehouse(id) {
    currentWarehouseId = parseInt(id);
    fetch(`${API_BASE_URL}/api/warehouses/switch/${id}`, { method: 'POST' })
        .then(() => {
            loadWarehouseConfig();
            updateVisualization();
        });
}

function loadWarehouseConfig() {
    fetch(`${API_BASE_URL}/api/warehouse/config?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(config => {
            warehouseConfig = config;
            renderWarehouse();
            updateVisualization(); // Reload items

            // Populate config inputs if view is visible
            document.getElementById('warehouse-length').value = config.length;
            document.getElementById('warehouse-width').value = config.width;
            document.getElementById('warehouse-height').value = config.height;
            // These inputs may not exist
            const gridSizeEl = document.getElementById('warehouse-grid-size');
            if (gridSizeEl) gridSizeEl.value = config.grid_size || 1;
            const levelsEl = document.getElementById('warehouse-levels');
            if (levelsEl) levelsEl.value = config.levels || 1;
            document.getElementById('warehouse-door-x').value = config.door_x || 0;
            document.getElementById('warehouse-door-y').value = config.door_y || 0;
            loadZones();
        });
}

// Helper for rectangular grids
function createRectangularGrid(L, W, step, color, opacity = 0.2, depthTest = true) {
    const points = [];
    const hL = L / 2;
    const hW = W / 2;
    // Ensure step is valid to prevent infinite loops
    if (step <= 0) step = 1;

    // X lines
    for (let z = -hW; z <= hW; z += step) {
        points.push(new THREE.Vector3(-hL, 0, z));
        points.push(new THREE.Vector3(hL, 0, z));
    }
    // Z lines
    for (let x = -hL; x <= hL; x += step) {
        points.push(new THREE.Vector3(x, 0, -hW));
        points.push(new THREE.Vector3(x, 0, hW));
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
        depthTest: depthTest
    });
    return new THREE.LineSegments(geometry, material);
}

function renderWarehouse() {
    // Clear old objects with proper disposal
    while (warehouseGroup.children.length > 0) {
        const obj = warehouseGroup.children[0];
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) {
                obj.material.forEach(m => m.dispose());
            } else {
                obj.material.dispose();
            }
        }
        warehouseGroup.remove(obj);
    }

    const { length, width, height, grid_size } = warehouseConfig;

    // Grid floor
    const gridHelper = createRectangularGrid(length, width, grid_size, 0x333333, 0.3);
    gridHelper.position.y = 0.01;
    warehouseGroup.add(gridHelper);

    // Floor plane
    const planeGeo = new THREE.PlaneGeometry(length, width);
    const planeMat = new THREE.MeshStandardMaterial({
        color: 0x0a0a0a, side: THREE.DoubleSide, roughness: 0.8
    });
    const floor = new THREE.Mesh(planeGeo, planeMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    floor.name = "warehouse_floor";
    warehouseGroup.add(floor);

    // Wireframe bounds
    const boxGeo = new THREE.BoxGeometry(length, height, width);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x00F0FF, opacity: 0.3, transparent: true }));
    line.position.y = height / 2;
    warehouseGroup.add(line);

    // Enhanced Door visualization
    const doorX = warehouseConfig.door_x || 0;
    const doorY = warehouseConfig.door_y || 0;
    const doorWidth = 2.5;
    const doorHeight = 3.0;
    const doorDepth = 0.5;

    // Determine wall orientation
    const distToLeft = doorX;
    const distToRight = length - doorX;
    const distToTop = doorY;
    const distToBottom = width - doorY;
    const minDist = Math.min(distToLeft, distToRight, distToTop, distToBottom);
    
    const isHorizontalWall = (minDist === distToTop || minDist === distToBottom);
    
    // Group for door frame components
    const doorGroup = new THREE.Group();
    
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x444444, metalness: 0.8, roughness: 0.2 });
    const postGeo = new THREE.BoxGeometry(0.3, doorHeight, doorDepth);
    const lintelGeo = new THREE.BoxGeometry(doorWidth + 0.6, 0.3, doorDepth);
    
    const leftPost = new THREE.Mesh(postGeo, frameMat);
    leftPost.position.set(-doorWidth/2, doorHeight/2, 0);
    const rightPost = new THREE.Mesh(postGeo, frameMat);
    rightPost.position.set(doorWidth/2, doorHeight/2, 0);
    const lintel = new THREE.Mesh(lintelGeo, frameMat);
    lintel.position.set(0, doorHeight + 0.15, 0);
    
    doorGroup.add(leftPost);
    doorGroup.add(rightPost);
    doorGroup.add(lintel);
    
    // Glowing field entrance
    const fieldGeo = new THREE.PlaneGeometry(doorWidth, doorHeight);
    const fieldMat = new THREE.MeshBasicMaterial({ 
        color: 0xFFD600, 
        transparent: true, 
        opacity: 0.3,
        side: THREE.DoubleSide
    });
    const field = new THREE.Mesh(fieldGeo, fieldMat);
    field.position.set(0, doorHeight/2, 0);
    doorGroup.add(field);
    
    doorGroup.position.set(doorX - length / 2, 0, doorY - width / 2);
    
    // Rotate based on wall
    if (!isHorizontalWall) {
        doorGroup.rotation.y = Math.PI / 2;
    }
    
    warehouseGroup.add(doorGroup);

    // Layer planes - removed per user request
    /*
    let layers = [];
    if (warehouseConfig.layer_heights && warehouseConfig.layer_heights.length > 0) {
        layers = warehouseConfig.layer_heights;
    } else {
        const lvls = warehouseConfig.levels || 1;
        const hPerLvl = height / lvls;
        for (let i = 0; i < lvls; i++) layers.push(hPerLvl);
    }

    let currentH = 0;
    // Skip ground floor (already drawn)
    for (let i = 0; i < layers.length; i++) {
        currentH += layers[i];
        if (currentH >= height) break; // Don't draw ceiling if it's top

    // Code removed to prevent global grid rendering
    }
    */

    renderGridLabels(length, width);
}

function renderGridLabels(length, width) {
    // Clear old labels
    coordinateLabels.forEach(l => scene.remove(l));
    coordinateLabels = [];

    function createLabel(text, pos) {
        const canvas = document.createElement('canvas');
        canvas.width = 128; // Increased
        canvas.height = 64; // Increased
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(0,0,0,0)'; // Transparent
        ctx.fillRect(0, 0, 128, 64);
        ctx.fillStyle = '#00F0FF';
        ctx.font = 'bold 24px monospace'; // Bigger
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 32); // Centered

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.copy(pos);
        sprite.scale.set(2, 1, 1);

        scene.add(sprite);
        coordinateLabels.push(sprite);
    }

    // X Axis Labels (Along Back Edge: Z = -width/2)
    for (let x = 0; x <= length; x += 2) {
        createLabel(x.toString(), new THREE.Vector3(x - length / 2, 0.5, -width / 2 - 1));
    }

    // Z Axis (Width) Labels (Along Left Edge: X = -length/2)
    for (let z = 0; z <= width; z += 2) {
        createLabel(z.toString(), new THREE.Vector3(-length / 2 - 1, 0.5, z - width / 2));
    }

    // Y Axis (Height) Labels (At Back-Left Corner)
    // Use warehouseConfig.height or just go up to reasonable amount
    const h = warehouseConfig.height || 5;
    for (let y = 0; y <= h; y += 1) { // Every 1m
        createLabel(y.toString() + 'm', new THREE.Vector3(-length / 2 - 1, y, -width / 2 - 1));
    }

}

function updateVisualization() {
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            items.forEach(i => allItemsData[i.id] = i);
            renderItems(items);
        });
}

// Helper for 6-axis rotation dimensions
function getRotatedDims(l, w, h, rotationCode) {
    const code = Math.round(rotationCode) % 6;
    switch (code) {
        case 0: return { dx: l, dy: w, dz: h };
        case 1: return { dx: w, dy: l, dz: h };
        case 2: return { dx: l, dy: h, dz: w };
        case 3: return { dx: h, dy: l, dz: w };
        case 4: return { dx: w, dy: h, dz: l };
        case 5: return { dx: h, dy: w, dz: l };
        default: return { dx: l, dy: w, dz: h };
    }
}

function renderItems(items) {
    // Dispose old objects to prevent memory leaks
    while (itemsGroup.children.length > 0) {
        const mesh = itemsGroup.children[0];
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach(m => m.dispose());
            } else {
                mesh.material.dispose();
            }
        }
        itemsGroup.remove(mesh);
    }
    if (pickerPathGroup) {
        while (pickerPathGroup.children.length > 0) {
            const obj = pickerPathGroup.children[0];
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
            pickerPathGroup.remove(obj);
        }
    }

    const mode = document.getElementById('color-mode') ? document.getElementById('color-mode').value : 'category';
    const showPath = document.getElementById('show-path') ? document.getElementById('show-path').checked : false;

    // Count overflow items (Z >= 1000 means doesn't fit)
    const OVERFLOW_Z_THRESHOLD = 1000;
    const overflowItems = items.filter(i => i.z >= OVERFLOW_Z_THRESHOLD);
    const fittingItems = items.filter(i => i.z < OVERFLOW_Z_THRESHOLD);

    // Update overflow indicator in UI
    let overflowIndicator = document.getElementById('overflow-indicator');
    if (!overflowIndicator) {
        // Create indicator if it doesn't exist
        const container = document.getElementById('progress-panel') || document.body;
        overflowIndicator = document.createElement('div');
        overflowIndicator.id = 'overflow-indicator';
        overflowIndicator.style.cssText = 'padding: 8px 12px; margin-top: 10px; border-radius: 6px; font-size: 0.85rem; display: none;';
        container.appendChild(overflowIndicator);
    }

    if (overflowItems.length > 0) {
        overflowIndicator.style.display = 'block';
        overflowIndicator.style.background = 'rgba(255, 0, 85, 0.2)';
        overflowIndicator.style.border = '1px solid #FF0055';
        overflowIndicator.style.color = '#FF6B8A';

        // Build list of unplaced items (show first 10, with expand option)
        const showCount = Math.min(overflowItems.length, 10);
        let itemsList = overflowItems.slice(0, showCount).map(item =>
            `<div style="padding: 2px 0; font-size: 0.75rem; color: #ccc;">
                • ${item.name || item.id} <span style="color:#888;">(${item.category || 'N/A'}) ${item.length}×${item.width}×${item.height}</span>
            </div>`
        ).join('');

        if (overflowItems.length > 10) {
            itemsList += `<div style="padding: 2px 0; font-size: 0.75rem; color: #888;">... and ${overflowItems.length - 10} more</div>`;
        }

        overflowIndicator.innerHTML = `
            <div style="cursor: pointer;" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none';">
                ⚠️ <strong>${overflowItems.length}</strong> items don't fit (${fittingItems.length}/${items.length} placed) 
                <span style="font-size: 0.7rem; color: #888;">▼ click to expand</span>
            </div>
            <div style="display: none; margin-top: 8px; max-height: 200px; overflow-y: auto;">
                ${itemsList}
            </div>
        `;
    } else {
        overflowIndicator.style.display = 'none';
    }

    // Calculate stats for heatmaps
    let maxWeight = 0;
    let maxAccess = 0;
    fittingItems.forEach(i => {
        if (i.weight > maxWeight) maxWeight = i.weight;
        if (i.access_freq > maxAccess) maxAccess = i.access_freq;
    });

    if (showPath) {
        renderPickerPath(fittingItems);
    }

    // Render fitting items only
    fittingItems.forEach(item => {
        // Apply 6-axis rotation
        const rotCode = item.rotation || 0;
        const dims = getRotatedDims(item.length, item.width, item.height, rotCode);

        // BoxGeometry(width, height, depth) = our (dx, dz, dy)
        const geometry = new THREE.BoxGeometry(dims.dx, dims.dz, dims.dy);
        const color = getItemColor(item, mode, maxWeight, maxAccess);
        const material = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.3,
            metalness: 0.1
        });

        const mesh = new THREE.Mesh(geometry, material);

        // Position: x,y are centers, z is bottom
        mesh.position.set(
            item.x - warehouseConfig.length / 2,
            item.z + dims.dz / 2,
            item.y - warehouseConfig.width / 2
        );

        // No extra rotation (baked into dimensions)
        mesh.userData = item;
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        itemsGroup.add(mesh);
    });
}

function renderPickerPath(items) {
    if (!items || items.length === 0) return;

    // Greedy TSP simulation for picking order
    const doorX = (warehouseConfig.door_x || 0);
    const doorY = (warehouseConfig.door_y || 0);
    const L = warehouseConfig.length;
    const W = warehouseConfig.width;

    let currentPos = { x: doorX, y: doorY };
    let remaining = [...items].map(i => ({
        ...i,
        cx: i.x,
        cy: i.y
    }));

    const points = [];
    // Start at door
    points.push(new THREE.Vector3(doorX - L / 2, 0.1, doorY - W / 2));

    while (remaining.length > 0) {
        let nearestIdx = -1;
        let minDst = Infinity;

        for (let i = 0; i < remaining.length; i++) {
            const item = remaining[i];
            const dst = Math.sqrt(Math.pow(item.cx - currentPos.x, 2) + Math.pow(item.cy - currentPos.y, 2));
            if (dst < minDst) {
                minDst = dst;
                nearestIdx = i;
            }
        }

        if (nearestIdx !== -1) {
            const nextItem = remaining[nearestIdx];
            // Add point
            points.push(new THREE.Vector3(nextItem.cx - L / 2, 0.5, nextItem.cy - W / 2)); // Path slightly elevated
            currentPos = { x: nextItem.cx, y: nextItem.cy };
            remaining.splice(nearestIdx, 1);
        } else {
            break;
        }
    }

    // Return to door? Optional. For now just path to last item.

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0xFFFF00, linewidth: 2 });
    const line = new THREE.Line(geometry, material);
    pickerPathGroup.add(line);
}

function getItemColor(item, mode, maxWeight, maxAccess) {
    if (mode === 'category') {
        // Hash string to color
        const str = item.category || 'General';
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }

        // Use HSL for pleasing colors
        const hue = Math.abs(hash % 360);
        const saturation = 70;
        const lightness = 50;

        return new THREE.Color(`hsl(${hue}, ${saturation}%, ${lightness}%)`);

    } else if (mode === 'weight') {
        // Green (light) -> Red (heavy)
        const t = maxWeight > 0 ? (item.weight / maxWeight) : 0;
        const color = new THREE.Color().setHSL(0.33 - (t * 0.33), 1, 0.5); // 0.33=Green, 0=Red
        return color;
    } else if (mode === 'access') {
        // Blue (low) -> Red (high)
        const t = maxAccess > 0 ? (item.access_freq / maxAccess) : 0;
        const color = new THREE.Color().setHSL(0.66 - (t * 0.66), 1, 0.5); // 0.66=Blue, 0=Red
        return color;
    } else if (mode === 'fragility') {
        // Fragile = Red, Robust = Green
        const isFragile = item.fragility === 1 || item.fragility === true;
        return isFragile ? new THREE.Color(0xFF0055) : new THREE.Color(0x00FF9D);
    }
    return 0x888888;
}

// Picker path controls

function togglePickerPath() {
    const showPath = document.getElementById('show-path').checked;
    const configPanel = document.getElementById('picker-path-config');

    if (showPath) {
        if (configPanel) configPanel.style.display = 'block';
    } else {
        if (configPanel) configPanel.style.display = 'none';
        clearPickerPath();
    }
}

function clearPickerPath() {
    while (pickerPathGroup.children.length > 0) {
        const obj = pickerPathGroup.children[0];
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
        pickerPathGroup.remove(obj);
    }
}

function generatePickerPath() {
    clearPickerPath();

    const itemCount = parseInt(document.getElementById('picker-item-count').value) || 5;
    const selectionMode = document.getElementById('picker-selection-mode').value;

    // Get placed items
    const items = Object.values(allItemsData).filter(i => i.z < 1000);
    if (items.length === 0) {
        alert('No items to create path for');
        return;
    }

    // Door position
    const doorX = warehouseConfig.door_x || 0;
    const doorY = warehouseConfig.door_y || 0;
    const whLength = warehouseConfig.length || 20;
    const whWidth = warehouseConfig.width || 20;

    // Select items based on mode
    let selectedItems = [];

    if (selectionMode === 'access') {
        // Sort by access frequency (highest first)
        selectedItems = [...items].sort((a, b) => (b.access_freq || 0) - (a.access_freq || 0)).slice(0, itemCount);
    } else if (selectionMode === 'random') {
        // Random selection
        const shuffled = [...items].sort(() => Math.random() - 0.5);
        selectedItems = shuffled.slice(0, itemCount);
    } else if (selectionMode === 'nearest') {
        // Sort by distance from door (orthogonal)
        selectedItems = [...items].sort((a, b) => {
            const distA = Math.abs(a.x - doorX) + Math.abs(a.y - doorY);
            const distB = Math.abs(b.x - doorX) + Math.abs(b.y - doorY);
            return distA - distB;
        }).slice(0, itemCount);
    }

    if (selectedItems.length === 0) return;

    // Simple greedy path from door
    const pathPoints = [];
    const startPoint = new THREE.Vector3(doorX - whLength / 2, 0.2, doorY - whWidth / 2);
    pathPoints.push(startPoint);

    const visited = new Set();
    let currentPos = { x: doorX, y: doorY };

    while (visited.size < selectedItems.length) {
        let nearestItem = null;
        let nearestDist = Infinity;

        for (const item of selectedItems) {
            if (visited.has(item.id)) continue;
            // Use orthogonal distance for realistic aisle movement
            const dist = Math.abs(item.x - currentPos.x) + Math.abs(item.y - currentPos.y);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearestItem = item;
            }
        }

        if (nearestItem) {
            visited.add(nearestItem.id);
            
            // Add orthogonal midpoint (L-shape path) to mimic aisle routing
            const midPoint = new THREE.Vector3(
                currentPos.x - whLength / 2, // Keep X same initially
                0.2, // close to floor
                nearestItem.y - whWidth / 2 // Move Y first
            );
            pathPoints.push(midPoint);
            
            const itemPoint = new THREE.Vector3(
                nearestItem.x - whLength / 2,
                (nearestItem.z || 0) + (nearestItem.height || 0.5) / 2,
                nearestItem.y - whWidth / 2
            );
            pathPoints.push(itemPoint);
            currentPos = { x: nearestItem.x, y: nearestItem.y };
        }
    }

    // Draw path line
    const geometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
    const material = new THREE.LineBasicMaterial({ color: 0xFFD600, linewidth: 3 });
    const pathLine = new THREE.Line(geometry, material);
    pickerPathGroup.add(pathLine);

    // Add markers at each stop
    for (let i = 1; i < pathPoints.length; i++) {
        const markerGeo = new THREE.SphereGeometry(0.15, 16, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0xFF0055 });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.copy(pathPoints[i]);
        pickerPathGroup.add(marker);

        // Add number label (simple sprite)
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#FFD600';
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(i.toString(), 32, 32);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.copy(pathPoints[i]);
        sprite.position.y += 0.5;
        sprite.scale.set(0.5, 0.5, 0.5);
        pickerPathGroup.add(sprite);
    }


}

// Optimization controls



function startOptimization() {
    const algo = document.getElementById('algorithm-select').value;

    const params = {
        warehouse_id: currentWarehouseId,
        weights: {
            space: parseFloat(document.getElementById('weight-space').value),
            accessibility: parseFloat(document.getElementById('weight-accessibility').value),
            stability: parseFloat(document.getElementById('weight-stability').value)
        }
    };

    // Load items before starting
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            items.forEach(i => allItemsData[i.id] = i);
            startOptimizationRequest(algo, params);
        })
        .catch(err => {
            alert(`Failed to load items: ${err}`);
        });
}

let currentOptimizationType = null;

function startOptimizationRequest(algo, params) {
    let endpoint = `/api/optimize/${algo}`;
    currentOptimizationType = algo;

    fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    }).then(res => res.json())
        .then(data => {
            if (data.success) {
                isOptimizing = true;
                const startBtn = document.getElementById('start-btn');
                const stopBtn = document.getElementById('stop-btn');
                const statusText = document.getElementById('status-text');
                const statusDot = document.getElementById('status-dot');

                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                if (statusText) statusText.textContent = "OPTIMIZING...";
                if (statusDot) statusDot.classList.add('active');

                startPolling();
            } else {
                alert(`Error starting optimization: ${data.error}`);
            }
        })
        .catch(err => {
            alert(`Network error: ${err}`);
        });
}

function stopOptimization() {
    let endpoint = '/api/optimize/stop';
    if (currentOptimizationType === 'compare') {
        endpoint = '/api/optimize/compare/stop';
    }

    fetch(`${API_BASE_URL}${endpoint}`, { method: 'POST' })
        .then(() => {
            isOptimizing = false;
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
        });
}

function startPolling() {
    // Clear any existing interval just in case
    if (optimizationInterval) clearInterval(optimizationInterval);

    optimizationInterval = setInterval(() => {
        let statusEndpoint = '/api/optimize/status';
        if (currentOptimizationType === 'compare') {
            statusEndpoint = '/api/optimize/compare/status';
        }

        fetch(`${API_BASE_URL}${statusEndpoint}`)
            .then(res => res.json())
            .then(status => {
                if (!status.running) {
                    clearInterval(optimizationInterval);
                    isOptimizing = false;

                    const startBtn = document.getElementById('start-btn');
                    const stopBtn = document.getElementById('stop-btn');
                    if (startBtn) startBtn.disabled = false;
                    if (stopBtn) stopBtn.disabled = true;

                    const statusText = document.getElementById('status-text');
                    const statusDot = document.getElementById('status-dot');
                    if (statusText) statusText.textContent = "COMPLETED";
                    if (statusDot) statusDot.classList.remove('active');

                    loadAnalytics(); // Refresh stats

                    if (currentOptimizationType !== 'compare') {
                        updateVisualization(); // Reload items from DB to ensure view matches final result
                    } else {
                        // For comparison, maybe show a summary modal or just the table update
                        document.querySelector('.view-panel[id="view-analytics"]').style.display = 'block';
                        switchView('analytics');
                        // Reset progress bar
                        const progBar = document.getElementById('progress-bar-fill');
                        if (progBar) progBar.style.width = '100%';
                        if (statusText) statusText.textContent = "COMPARISON DONE";
                    }
                }

                // Update UI
                const progress = status.progress || 0;
                const progPct = document.getElementById('progress-percent');
                const progBar = document.getElementById('progress-bar-fill');
                const bestFit = document.getElementById('best-fitness');

                if (progPct) progPct.textContent = Math.round(progress) + '%';
                if (progBar) progBar.style.width = progress + '%';
                // For compare, this might be misleading if it's jumping between algos

                // Special handling for comparison status
                if (currentOptimizationType === 'compare') {
                    if (bestFit) bestFit.textContent = '-'; // Don't show global best fit for comparison
                } else {
                    if (bestFit) bestFit.textContent = (status.best_fitness || 0).toFixed(4);
                }

                // Update Status Text with detailed message
                const statusText = document.getElementById('status-text');
                if (statusText) {
                    if (status.message) {
                        statusText.textContent = status.message;
                        statusText.title = status.message;
                    } else {
                        statusText.textContent = "OPTIMIZING...";
                    }
                }

                // Update Extended Status Info
                const statusAlgo = document.getElementById('status-algo');
                const statusGen = document.getElementById('status-gen');
                const statusPlaced = document.getElementById('status-placed');
                const statusElapsed = document.getElementById('status-elapsed');

                if (statusAlgo) {
                    if (currentOptimizationType === 'compare') {
                        statusAlgo.textContent = status.current_algorithm || 'Comparing...';
                    } else {
                        statusAlgo.textContent = status.algorithm || document.getElementById('algorithm-select')?.value?.toUpperCase() || '-';
                    }
                }

                if (statusGen) {
                    if (currentOptimizationType === 'compare') {
                        statusGen.textContent = `${status.current_algorithm_index || 0}/${status.total_algorithms || 4}`;
                    } else {
                        const gen = status.generation || status.iteration || Math.round(progress);
                        const total = status.total_generations || status.total_iterations || 100;
                        statusGen.textContent = `${gen}/${total}`;
                    }
                }

                // Visual updates only for single mode
                if (currentOptimizationType !== 'compare') {
                    if (statusPlaced && status.best_solution) {
                        const placed = status.best_solution.filter(s => s.z < 1000).length;
                        const total = status.best_solution.length;
                        statusPlaced.textContent = `${placed}/${total}`;
                    }
                    if (statusElapsed && status.elapsed_time) {
                        statusElapsed.textContent = status.elapsed_time.toFixed(1) + 's';
                    }

                    if (status.best_solution && status.best_solution.length > 0) {
                        // Map solution coordinates back to full item data
                        const solutionItems = status.best_solution.map(sol => {
                            const originalItem = allItemsData[sol.id];
                            if (originalItem) {
                                return { ...originalItem, ...sol };
                            }
                            return null;
                        }).filter(item => item !== null);

                        // Only re-render if we have items to show (prevents blank screen)
                        if (solutionItems.length > 0) {
                            renderItems(solutionItems);
                        }
                    }
                }
            });
    }, 200); // Faster polling for smooth updates
}

// Analytics
function loadAnalytics() {
    fetch(`${API_BASE_URL}/api/metrics/current?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            const mSpace = document.getElementById('metric-space');
            const mFreeSpace = document.getElementById('metric-free-space');
            const mRuntime = document.getElementById('metric-runtime');
            const mAccess = document.getElementById('metric-access');
            const mStab = document.getElementById('metric-stability');
            const mCount = document.getElementById('metric-count');

            if (mSpace) mSpace.textContent = (data.space_utilization * 100).toFixed(1) + '%';
            if (mFreeSpace) mFreeSpace.textContent = data.free_space_vol ? data.free_space_vol.toFixed(2) + ' m³' : '0.00 m³';
            if (mRuntime) mRuntime.textContent = data.execution_time ? data.execution_time.toFixed(2) + 's' : '0.00s';
            if (mAccess) mAccess.textContent = data.accessibility.toFixed(2);
            if (mStab) mStab.textContent = (data.stability * 100).toFixed(1) + '%';
            if (mCount) mCount.textContent = data.total_items;

            // Also update rendered count
            countRenderedBoxes();
        });

    fetch(`${API_BASE_URL}/api/metrics/categories?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            initCategoryChart(data);
        });

    // Fetch Algorithm Best Performance
    fetch(`${API_BASE_URL}/api/metrics/algo-best?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            const algoBestCards = document.getElementById('algo-best-cards');
            if (algoBestCards) {
                algoBestCards.innerHTML = '';

                if (data.length === 0) {
                    algoBestCards.innerHTML = '<div style="padding:10px; color:var(--text-muted); text-align:center;">No optimization runs yet</div>';
                    return;
                }

                // Pre-compute max values for colour coding
                const maxSpace = Math.max(...data.map(d => d.space_utilization || 0));
                const maxAccess = Math.max(...data.map(d => d.accessibility || 0));
                const maxStab = Math.max(...data.map(d => d.stability || 0));
                const maxGroup = Math.max(...data.map(d => d.grouping || 0));
                
                const goodColor = (v, max) => {
                    if (!v || !max) return 'var(--text-muted)';
                    const t = v / max;
                    return t >= 0.9 ? 'var(--success)' : t >= 0.6 ? 'var(--text-main)' : 'var(--warning)';
                };
                
                const pct = (v) => (v != null && v > 0) ? (v * 100).toFixed(1) + '%' : '-';
                const dec = (v) => (v != null && v > 0) ? v.toFixed(3) : '-';
                const sec = (v) => (v != null && v > 0) ? v.toFixed(2) + 's' : '-';

                data.forEach(algo => {
                    // Format algorithm name
                    let algoName = algo.algorithm;
                    let algoColor = '#fff';
                    if (algoName.includes('Genetic') && !algoName.includes('Hybrid')) {
                        algoName = 'GA';
                        algoColor = '#00f0ff';
                    } else if (algoName.includes('Extremal') && !algoName.includes('Hybrid')) {
                        algoName = 'EO';
                        algoColor = '#7000ff';
                    } else if (algoName.includes('GA+EO') || algoName.includes('GA-EO')) {
                        algoName = 'GA+EO';
                        algoColor = '#ff0055';
                    } else if (algoName.includes('EO+GA') || algoName.includes('EO-GA')) {
                        algoName = 'EO+GA';
                        algoColor = '#FFD600';
                    }

                    // Format timestamp
                    let achievedAt = '-';
                    if (algo.timestamp) {
                        const date = new Date(algo.timestamp);
                        achievedAt = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    }

                    const card = document.createElement('div');
                    card.className = 'algo-performance-card';
                    card.innerHTML = `
                        <div class="algo-card-header">
                            <span class="algo-card-name" style="color:${algoColor}">${algoName}</span>
                            <span class="algo-card-fitness" title="Best Fitness">${algo.best_fitness ? algo.best_fitness.toFixed(4) : '0.0000'}</span>
                        </div>
                        <div class="algo-card-metrics">
                            <div class="mini-metric">
                                <span class="mini-label">Space</span>
                                <span class="mini-value" style="color:${goodColor(algo.space_utilization, maxSpace)}">${pct(algo.space_utilization)}</span>
                            </div>
                            <div class="mini-metric">
                                <span class="mini-label">Access</span>
                                <span class="mini-value" style="color:${goodColor(algo.accessibility, maxAccess)}">${dec(algo.accessibility)}</span>
                            </div>
                            <div class="mini-metric">
                                <span class="mini-label">Stable</span>
                                <span class="mini-value" style="color:${goodColor(algo.stability, maxStab)}">${pct(algo.stability)}</span>
                            </div>
                            <div class="mini-metric">
                                <span class="mini-label">Group</span>
                                <span class="mini-value" style="color:${goodColor(algo.grouping, maxGroup)}">${dec(algo.grouping)}</span>
                            </div>
                            <div class="mini-metric">
                                <span class="mini-label">T-Best</span>
                                <span class="mini-value" style="color:var(--accent-primary)">${sec(algo.time_to_best)}</span>
                            </div>
                            <div class="mini-metric">
                                <span class="mini-label">Total</span>
                                <span class="mini-value" style="color:var(--text-muted)">${sec(algo.execution_time)}</span>
                            </div>
                        </div>
                        <div class="algo-card-footer">${achievedAt}</div>
                    `;
                    algoBestCards.appendChild(card);
                });
            }
        });

    fetch(`${API_BASE_URL}/api/metrics/history?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            initHistoryChart(data);

            // Populate History Log Table
            const historyBody = document.getElementById('analytics-history-body');
            if (historyBody) {
                historyBody.innerHTML = '';
                data.forEach(run => {
                    const row = document.createElement('tr');
                    row.style.borderBottom = '1px solid #333';

                    let algoName = run.algorithm.replace('_COMPARE', ' (C)');
                    if (algoName === 'Genetic Algorithm') algoName = 'GA';
                    if (algoName === 'Extremal Optimization') algoName = 'EO';

                    row.innerHTML = `
                        <td style="padding:6px; font-weight:bold; color:var(--text-main);">${algoName}</td>
                        <td style="padding:6px;">${run.fitness ? run.fitness.toFixed(1) : '0.0'}</td>
                        <td style="padding:6px; color:#00f0ff;">${run.time_to_best ? run.time_to_best.toFixed(2) + 's' : '-'}</td>
                        <td style="padding:6px;">${run.execution_time ? run.execution_time.toFixed(2) + 's' : '-'}</td>
                    `;
                    historyBody.appendChild(row);
                });
            }
        });
}

function initHistoryChart(history) {
    const ctx = document.getElementById('historyChart');
    if (!ctx) return;
    if (historyChart) historyChart.destroy();

    // Last 10 runs
    const data = history.slice(-10);
    const labels = data.map(d => new Date(d.timestamp).toLocaleTimeString());
    const fitness = data.map(d => d.fitness);

    historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fitness Score',
                data: fitness,
                borderColor: '#00F0FF',
                backgroundColor: 'rgba(0, 240, 255, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8899A6' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8899A6' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8899A6' } }
            }
        }
    });
}

function initCategoryChart(data) {
    const ctx = document.getElementById('categoryChart');
    if (!ctx) return;

    if (categoryChart) categoryChart.destroy();

    categoryChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.categories,
            datasets: [{
                data: data.counts,
                backgroundColor: ['#00F0FF', '#7000FF', '#FF0055', '#FFD600', '#FFFFFF'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#8899A6' } }
            }
        }
    });
}

// Item list
function loadItemsList() {
    const container = document.getElementById('items-list');
    container.innerHTML = 'Loading...';

    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            container.innerHTML = '';
            const table = document.createElement('table');
            table.className = 'compact-table';
            table.style.marginTop = '8px';

            const thead = document.createElement('thead');
            thead.innerHTML = `
                <tr style="color:var(--accent-primary); border-bottom:1px solid #444;">
                    <th style="width: 40%;">Name</th>
                    <th style="width: 30%;">Dim</th>
                    <th style="width: 20%;">Cat</th>
                    <th style="width: 10%;">F</th>
                </tr>
            `;
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            items.forEach(item => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                tr.innerHTML = `
                    <td title="${item.name || item.id}">${item.name || item.id}</td>
                    <td style="color: var(--accent-primary)">${item.width}x${item.height}x${item.length}</td>
                    <td>${item.category}</td>
                    <td>${item.fragility ? '⚠️' : '🛡️'}</td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            container.appendChild(table);
        });
}

let editingZoneId = null;
let currentZones = [];

function loadZones() {
    fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            currentZones = zones; // Store locally
            renderZones(zones);
            updateZonesList(zones);
        });
}






function editZone(id) {
    const zone = currentZones.find(z => z.id === id);
    if (!zone) return;

    document.getElementById('zone-name').value = zone.name;
    document.getElementById('zone-x1').value = zone.x1;
    document.getElementById('zone-y1').value = zone.y1;
    document.getElementById('zone-x2').value = zone.x2;
    document.getElementById('zone-y2').value = zone.y2;
    document.getElementById('zone-z1').value = zone.z1 || 0;
    document.getElementById('zone-z2').value = zone.z2 || '';
    document.getElementById('zone-type').value = zone.zone_type;

    // Load metadata
    if (zone.metadata) {
        if (zone.metadata.layer_heights) {
            document.getElementById('zone-layer-heights').value = zone.metadata.layer_heights.join(', ');
        } else {
            document.getElementById('zone-layer-heights').value = '';
        }

        if (zone.metadata.levels) {
            const levelInput = document.getElementById('zone-levels');
            if (levelInput) levelInput.value = zone.metadata.levels;
        } else {
            const levelInput = document.getElementById('zone-levels');
            if (levelInput) levelInput.value = '';
        }
    } else {
        document.getElementById('zone-layer-heights').value = '';
        const levelInput = document.getElementById('zone-levels');
        if (levelInput) levelInput.value = '';
    }

    editingZoneId = id;
    const btn = document.querySelector('#view-warehouse button[onclick="addZone()"]');
    if (btn) btn.innerText = "UPDATE ZONE";
}



function renderZones(zones) {
    // Clear old zones with disposal
    function disposeObject(obj) {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) {
                obj.material.forEach(m => m.dispose());
            } else {
                obj.material.dispose();
            }
        }
        // Dispose children
        while (obj.children && obj.children.length > 0) {
            disposeObject(obj.children[0]);
            obj.remove(obj.children[0]);
        }
    }
    while (exclusionZonesGroup.children.length > 0) {
        disposeObject(exclusionZonesGroup.children[0]);
        exclusionZonesGroup.remove(exclusionZonesGroup.children[0]);
    }

    zones.forEach(zone => {
        const width = zone.x2 - zone.x1;
        const depth = zone.y2 - zone.y1;

        // Z bounds
        const whHeight = warehouseConfig ? warehouseConfig.height : 5;
        let z1 = zone.z1 !== undefined ? zone.z1 : 0;
        let z2 = zone.z2 !== undefined ? zone.z2 : whHeight;
        // Clamp z2 if default 100
        if (z2 === 100 && whHeight < 100) z2 = whHeight;

        const zoneHeight = z2 - z1;
        if (zoneHeight <= 0) return; // Skip invalid

        const geometry = new THREE.BoxGeometry(width, zoneHeight, depth);
        const isAlloc = zone.zone_type === 'allocation';
        const color = isAlloc ? 0x00FF9D : 0xFF0055;

        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.3, // Increased from 0.15 for better visibility
            side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(geometry, material);

        // Alignment
        const whLength = warehouseConfig ? warehouseConfig.length : 20;
        const whWidth = warehouseConfig ? warehouseConfig.width : 20;

        mesh.position.set(
            (zone.x1 + width / 2) - whLength / 2,
            z1 + zoneHeight / 2,
            (zone.y1 + depth / 2) - whWidth / 2
        );

        // Wireframe
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
            color: color,
            depthTest: false, // Always show on top of other objects
            lights: false
        }));
        line.renderOrder = 1; // Try to force render on top
        mesh.add(line);

        // Shelf layers
        if (isAlloc) {
            let layerHeights = [];
            if (zone.metadata && zone.metadata.layer_heights && zone.metadata.layer_heights.length > 0) {
                layerHeights = zone.metadata.layer_heights;
            } else {
                // Default to 1 layer if not specified
                const levels = (zone.metadata && zone.metadata.levels) ? parseInt(zone.metadata.levels) : 1;

                if (levels > 1) {
                    const h = zoneHeight / levels;
                    for (let i = 0; i < levels; i++) layerHeights.push(h);
                }
            }

            // Draw shelf planes
            let currentY = -zoneHeight / 2;
            const hPerLayer = zoneHeight / layerHeights.length;

            for (let i = 0; i < layerHeights.length; i++) {
                const h = layerHeights[i];
                currentY += h;

                // Skip top face
                if (currentY >= zoneHeight / 2 - 0.01) break;

                // Shelf plane
                const shelfGeo = new THREE.PlaneGeometry(width, depth);
                const shelfMat = new THREE.MeshBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.6, // Increased opacity for visibility
                    side: THREE.DoubleSide
                });
                const shelf = new THREE.Mesh(shelfGeo, shelfMat);
                shelf.rotation.x = Math.PI / 2;
                shelf.position.y = currentY;

                const shelfEdges = new THREE.EdgesGeometry(shelfGeo);
                const shelfLine = new THREE.LineSegments(shelfEdges, new THREE.LineBasicMaterial({ color: 0xFFFFFF }));
                shelf.add(shelfLine);

                mesh.add(shelf);

                // Grid on shelf
                const gridSize = warehouseConfig.grid_size || 1;
                const layerGrid = createRectangularGrid(width, depth, gridSize, 0xFFFFFF, 1.0, false);
                layerGrid.position.y = currentY + 0.01;
                layerGrid.renderOrder = 2;
                mesh.add(layerGrid);
            }
        }

        exclusionZonesGroup.add(mesh);
    });
}

function updateZonesList(zones) {
    const container = document.getElementById('zones-list');
    if (!container) return;

    // Calculate item counts per zone from allItemsData
    const itemCounts = {};
    zones.forEach(z => {
        if (z.zone_type === 'allocation') {
            let count = 0;
            Object.values(allItemsData).forEach(item => {
                // Check if item center is within zone bounds
                if (item.x >= z.x1 && item.x <= z.x2 &&
                    item.y >= z.y1 && item.y <= z.y2 &&
                    (item.z || 0) >= (z.z1 || 0) &&
                    (item.z || 0) < (z.z2 || 999)) {
                    count++;
                }
            });
            itemCounts[z.id] = count;
        }
    });

    container.innerHTML = zones.map(z => {
        const isAlloc = z.zone_type === 'allocation';
        const layerInfo = (z.metadata && z.metadata.layer_heights && z.metadata.layer_heights.length > 0)
            ? `(Custom Layers)`
            : ((z.metadata && z.metadata.levels)
                ? `(Layers: ${z.metadata.levels})`
                : '');

        // Show item count for allocation zones
        const itemCountStr = isAlloc ? `<span style="color: var(--accent-primary); font-weight: 600;">📦 ${itemCounts[z.id] || 0} items</span>` : '';

        return `
        <div class="data-card" style="margin-bottom: 8px; padding: 12px; display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;">
                <div style="min-width: 0; flex: 1;">
                    <div style="font-weight: bold; color: ${isAlloc ? 'var(--success)' : 'var(--danger)'}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${z.name}">
                        ${z.name}
                    </div>
                    <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 2px;">
                        ${z.x1}-${z.x2}m × ${z.y1}-${z.y2}m × ${z.z1 || 0}-${z.z2 || 'Max'}m
                        ${layerInfo}
                    </div>
                </div>
                <div style="display: flex; gap: 4px; flex-shrink: 0;">
                     <button class="btn" style="padding: 4px 6px; font-size: 0.65rem; background: var(--accent-primary); min-width: 40px;" onclick="editZone(${z.id})">EDIT</button>
                     <button class="btn btn-danger" style="padding: 4px 6px; font-size: 0.65rem; min-width: 40px;" onclick="deleteZone(${z.id})">DEL</button>
                </div>
            </div>
            ${isAlloc ? `<div style="font-size: 0.7rem; color: var(--accent-primary); font-weight: 600; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4px;">📦 ${itemCounts[z.id] || 0} items</div>` : ''}
        </div>
    `}).join('');
}

function addZone() {
    const name = document.getElementById('zone-name').value;

    // Read Position & Size Inputs
    const posX = parseFloat(document.getElementById('zone-pos-x').value);
    const posY = parseFloat(document.getElementById('zone-pos-y').value);
    const width = parseFloat(document.getElementById('zone-width').value);
    const depth = parseFloat(document.getElementById('zone-depth').value);

    // Read Vertical Inputs
    const baseZ = parseFloat(document.getElementById('zone-base-z').value) || 0;
    const height = parseFloat(document.getElementById('zone-height').value) || 5;

    // Convert to Backend Format (x1, y1, x2, y2)
    const x1 = posX;
    const y1 = posY;
    const x2 = posX + width;
    const y2 = posY + depth;
    const z1 = baseZ;
    const z2 = baseZ + height;

    const type = document.getElementById('zone-type').value;
    const layerStr = document.getElementById('zone-layer-heights').value;
    const levelsVal = document.getElementById('zone-levels') ? parseInt(document.getElementById('zone-levels').value) : null;

    let layerHeights = [];
    if (layerStr) {
        layerHeights = layerStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    }

    if (!name || isNaN(x1) || isNaN(y1) || isNaN(width) || isNaN(depth)) {
        alert("Please fill name, position, and dimensions");
        return;
    }

    const payload = {
        warehouse_id: currentWarehouseId,
        name: name,
        x1: x1, y1: y1,
        z1: z1, z2: z2,
        x2: x2, y2: y2,
        zone_type: type,
        metadata: {
            layer_heights: layerHeights,
            levels: levelsVal
        }
    };

    let url = `${API_BASE_URL}/api/warehouse/zones`;
    let method = 'POST';

    // Check if we are updating (editingZoneId is set)
    if (editingZoneId) {
        url = `${API_BASE_URL}/api/warehouse/zones/${editingZoneId}`;
        method = 'PUT';
    }

    fetch(url, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).then(res => res.json())
        .then(data => {
            if (data.success) {
                loadZones();
                // Clear inputs
                document.getElementById('zone-name').value = '';
                document.getElementById('zone-pos-x').value = '';
                document.getElementById('zone-pos-y').value = '';
                document.getElementById('zone-width').value = '';
                document.getElementById('zone-depth').value = '';
                document.getElementById('zone-base-z').value = '0';
                document.getElementById('zone-height').value = '5';
                document.getElementById('zone-layer-heights').value = '';
                const levelInput = document.getElementById('zone-levels');
                if (levelInput) levelInput.value = '1';

                // Reset Edit Mode
                editingZoneId = null;
                const btn = document.querySelector('#view-warehouse button[onclick="addZone()"]');
                if (btn) btn.innerText = "ADD ZONE";
            }
        });
}

function editZone(id) {
    fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            const z = zones.find(zn => zn.id === id);
            if (!z) return;

            document.getElementById('zone-name').value = z.name;

            // Convert x1/x2 -> Pos/Size
            const width = z.x2 - z.x1;
            const depth = z.y2 - z.y1;
            const height = (z.z2 || 5) - (z.z1 || 0);

            document.getElementById('zone-pos-x').value = z.x1;
            document.getElementById('zone-pos-y').value = z.y1;
            document.getElementById('zone-width').value = width;
            document.getElementById('zone-depth').value = depth;

            document.getElementById('zone-base-z').value = z.z1 || 0;
            document.getElementById('zone-height').value = height;

            document.getElementById('zone-type').value = z.zone_type;

            const layers = (z.metadata && z.metadata.layer_heights) ? z.metadata.layer_heights.join(', ') : '';
            document.getElementById('zone-layer-heights').value = layers;

            const levels = (z.metadata && z.metadata.levels) ? z.metadata.levels : 1;
            const levelInput = document.getElementById('zone-levels');
            if (levelInput) levelInput.value = levels;

            editingZoneId = id;
            const btn = document.querySelector('#view-warehouse button[onclick="addZone()"]');
            if (btn) btn.innerText = "UPDATE ZONE";
        });
}

function deleteZone(id) {
    if (!confirm('Delete this zone?')) return;

    fetch(`${API_BASE_URL}/api/warehouse/zones/${id}?warehouse_id=${currentWarehouseId}`, { method: 'DELETE' })
        .then(res => res.json())
        .then(data => {

            if (data.success) {
                loadZones();
            } else {
                alert('Failed to delete zone: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(err => {
            console.error('Delete error:', err);
            alert('Error deleting zone: ' + err);
        });
}


function applyPreset(name) {
    if (!confirm('This will overwrite current warehouse configuration and zones. Continue?')) return;



    if (name === '4-shelves') {
        // 1. Update Config - Compact Physics Warehouse
        const newConfig = {
            name: "Physics Alloc (Compact)",
            length: 10,
            width: 10,
            height: 6,
            levels: 2,
            grid_size: 1,
            id: currentWarehouseId
        };



        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        }).then(res => res.json())
            .then(d => {

                if (d.success) {
                    // 2. Clear Zones & Add New Zones 
                    clearZones().then(() => {
                        const zones = [];
                        // Compact Layout: 10x10
                        // 4 Stacks of 3x3m
                        // Gap: 2m
                        // Shelf A (Top-Left): X:1-4, Y:1-4
                        // Shelf B (Top-Right): X:6-9, Y:1-4
                        // Shelf C (Bottom-Left): X:1-4, Y:6-9
                        // Shelf D (Bottom-Right): X:6-9, Y:6-9

                        const stacks = [
                            { name: 'Shelf A', x1: 1, y1: 1, x2: 4, y2: 4 },
                            { name: 'Shelf B', x1: 6, y1: 1, x2: 9, y2: 4 },
                            { name: 'Shelf C', x1: 1, y1: 6, x2: 4, y2: 9 },
                            { name: 'Shelf D', x1: 6, y1: 6, x2: 9, y2: 9 }
                        ];

                        stacks.forEach(s => {
                            // Bottom Zone (Heavy) - Z: 0-3
                            zones.push({
                                name: `${s.name} (Bottom)`,
                                x1: s.x1, y1: s.y1, x2: s.x2, y2: s.y2,
                                z1: 0, z2: 3,
                                zone_type: 'allocation',
                                metadata: { levels: 1 }
                            });
                            // Top Zone (Fragile) - Z: 3-6
                            zones.push({
                                name: `${s.name} (Top)`,
                                x1: s.x1, y1: s.y1, x2: s.x2, y2: s.y2,
                                z1: 3, z2: 6,
                                zone_type: 'allocation',
                                metadata: { levels: 1 }
                            });
                        });

                        const promises = zones.map(z =>
                            fetch(`${API_BASE_URL}/api/warehouse/zones`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId })
                            })
                        );

                        Promise.all(promises).then(() => {
                            loadWarehouseConfig();
                            alert('Preset Applied: 4 Stacks (8 Zones)');
                        });
                    });
                }
            });
    } else if (name === 'small-room') {
        const newConfig = {
            name: "Small Room", length: 5, width: 5, height: 3, levels: 2, grid_size: 0.5,
            door_x: 2.5, door_y: 0, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(() => {
            clearZones().then(() => {
                const zones = [];
                const racks = [
                    { name: 'Left Rack', x1: 0.5, y1: 0.5, x2: 1.5, y2: 4.5 },
                    { name: 'Right Rack', x1: 3.5, y1: 0.5, x2: 4.5, y2: 4.5 }
                ];
                racks.forEach(r => {
                    zones.push({ ...r, name: `${r.name} (Bottom)`, z1: 0, z2: 1.5, zone_type: 'allocation', metadata: { levels: 1 } });
                    zones.push({ ...r, name: `${r.name} (Top)`, z1: 1.5, z2: 3, zone_type: 'allocation', metadata: { levels: 1 } });
                });
                const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                Promise.all(promises).then(() => { loadWarehouseConfig(); alert('Applied: Small Room (5x5m, 4 Zones)'); });
            });
        });
    } else if (name === 'long-hall') {
        const newConfig = {
            name: "Long Hall", length: 30, width: 5, height: 4, levels: 2, grid_size: 1,
            door_x: 0, door_y: 2.5, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(() => {
            clearZones().then(() => {
                const zones = [];
                const racks = [
                    { name: 'Left Wall', x1: 1, y1: 0.5, x2: 29, y2: 1.5 },
                    { name: 'Right Wall', x1: 1, y1: 3.5, x2: 29, y2: 4.5 }
                ];
                racks.forEach(r => {
                    zones.push({ ...r, name: `${r.name} (Bottom)`, z1: 0, z2: 2, zone_type: 'allocation', metadata: { levels: 1 } });
                    zones.push({ ...r, name: `${r.name} (Top)`, z1: 2, z2: 4, zone_type: 'allocation', metadata: { levels: 1 } });
                });
                const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                Promise.all(promises).then(() => { loadWarehouseConfig(); alert('Applied: Long Hall (30x5m, 4 Zones)'); });
            });
        });
    } else if (name === 'massive-center') {
        const newConfig = {
            name: "Massive Center", length: 50, width: 50, height: 10, levels: 3, grid_size: 2,
            door_x: 25, door_y: 0, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(() => {
            clearZones().then(() => {
                const zones = [];
                const blocks = [
                    { name: 'Block A (NW)', x1: 5, y1: 5, x2: 20, y2: 20 },
                    { name: 'Block B (NE)', x1: 30, y1: 5, x2: 45, y2: 20 },
                    { name: 'Block C (SW)', x1: 5, y1: 30, x2: 20, y2: 45 },
                    { name: 'Block D (SE)', x1: 30, y1: 30, x2: 45, y2: 45 }
                ];
                blocks.forEach(b => {
                    zones.push({ ...b, name: `${b.name} (Floor)`, z1: 0, z2: 3, zone_type: 'allocation', metadata: { levels: 1 } });
                    zones.push({ ...b, name: `${b.name} (Mid)`, z1: 3, z2: 6, zone_type: 'allocation', metadata: { levels: 1 } });
                    zones.push({ ...b, name: `${b.name} (High)`, z1: 6, z2: 9, zone_type: 'allocation', metadata: { levels: 1 } });
                });
                const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                Promise.all(promises).then(() => { loadWarehouseConfig(); alert('Applied: Massive Center (50x50m, 12 Zones)'); });
            });
        });
    } else if (name === 'l-shape') {
        const newConfig = {
            name: "L-Shape Warehouse", length: 20, width: 20, height: 5, levels: 2, grid_size: 1,
            door_x: 10, door_y: 0, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(res => res.json()).then(d => {
            if (d.success) {
                clearZones().then(() => {
                    const zones = [];
                    // The L-Shape Cutout Exclusion Zone
                    zones.push({
                        name: "L-Shape Cutout", x1: 10, y1: 10, x2: 20, y2: 20,
                        z1: 0, z2: 5, zone_type: 'exclusion', metadata: {}
                    });
                    
                    const racks = [
                        { name: 'Top Arm Rack', x1: 2, y1: 2, x2: 8, y2: 18 },
                        { name: 'Right Arm Rack', x1: 12, y1: 2, x2: 18, y2: 8 }
                    ];
                    racks.forEach(r => {
                        zones.push({ ...r, name: `${r.name} (Lower)`, z1: 0, z2: 2.5, zone_type: 'allocation', metadata: { levels: 1 } });
                        zones.push({ ...r, name: `${r.name} (Upper)`, z1: 2.5, z2: 5, zone_type: 'allocation', metadata: { levels: 1 } });
                    });

                    const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                    Promise.all(promises).then(() => { loadWarehouseConfig(); alert('Applied: L-Shape Warehouse (Exclusion + 4 Allocation Zones)'); });
                });
            }
        });
    } else if (name === 'clear') {
        clearZones().then(() => {
            // Also reset config to default
            const defaultConfig = {
                name: "Warehouse " + currentWarehouseId,
                length: 20,
                width: 20,
                height: 5,
                levels: 1,
                grid_size: 1,
                door_x: 0,
                door_y: 0,
                id: currentWarehouseId
            };

            fetch(`${API_BASE_URL}/api/warehouse/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(defaultConfig)
            }).then(() => {
                loadWarehouseConfig();
                loadZones();
                alert('Cleared: Zones removed and configuration reset.');
            });
        });
    }
}

function deleteAllItems() {
    if (!confirm('Are you sure you want to delete ALL items? This cannot be undone.')) {
        return;
    }

    fetch(`${API_BASE_URL}/api/items/delete_all?warehouse_id=${currentWarehouseId}`, {
        method: 'DELETE'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadItems();
                updateVisualization();
                alert('All items deleted successfully.');
            } else {
                alert('Error deleting items: ' + data.error);
            }
        })
        .catch(error => console.error('Error:', error));
}

function scrambleItems() {
    if (!confirm('This will DELETE all current items and generate 50 RANDOM ones. Continue?')) {
        return;
    }

    const btn = event.srcElement; // Get button to show loading state
    const originalText = btn.textContent;
    btn.textContent = 'SCRAMBLING...';
    btn.disabled = true;

    fetch(`${API_BASE_URL}/api/items/scramble`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ count: 50 })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadItems();
                updateVisualization();
                // alert('Values Scrambled!'); 
            } else {
                alert('Error scrambling: ' + data.message);
            }
        })
        .catch(error => console.error('Error:', error))
        .finally(() => {
            btn.textContent = originalText;
            btn.disabled = false;
        });
}


function clearZones() {

    return fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            if (!Array.isArray(zones)) {
                console.error('Expected array of zones, got:', zones);
                throw new Error('Invalid response from server when fetching zones.');
            }

            const promises = zones.map(z =>
                fetch(`${API_BASE_URL}/api/warehouse/zones/${z.id}?warehouse_id=${currentWarehouseId}`, { method: 'DELETE' })
            );
            return Promise.all(promises);
        })
        .catch(err => {
            console.error('Error clearing zones:', err);
            alert('Error clearing zones: ' + err.message);
            throw err; // Re-throw to stop chain if needed
        });
}


function updateWarehouseConfig() {
    const data = {
        name: "Warehouse " + currentWarehouseId,
        length: parseFloat(document.getElementById('warehouse-length')?.value || 20),
        width: parseFloat(document.getElementById('warehouse-width')?.value || 20),
        height: parseFloat(document.getElementById('warehouse-height')?.value || 5),
        grid_size: parseFloat(document.getElementById('warehouse-grid-size')?.value || 1),
        levels: parseInt(document.getElementById('warehouse-levels')?.value || 1),
        door_x: parseFloat(document.getElementById('warehouse-door-x')?.value || 0),
        door_y: parseFloat(document.getElementById('warehouse-door-y')?.value || 0),
        id: currentWarehouseId
    };

    fetch(`${API_BASE_URL}/api/warehouse/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }).then(res => res.json())
        .then(d => {
            if (d.success) {
                loadWarehouseConfig();
                alert('Configuration Saved');
            }
        });
}

function uploadCSV() {
    const input = document.getElementById('file-upload');
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch(`${API_BASE_URL}/api/upload-csv?warehouse_id=${currentWarehouseId}`, {
        method: 'POST',
        body: formData
    }).then(res => res.json())
        .then(d => {
            if (d.success) {
                alert('Uploaded Successfully');
                updateVisualization();
                if (document.getElementById('view-items').style.display === 'block') loadItemsList();
            } else {
                alert('Error: ' + d.error);
            }
        });
}

let comparisonInterval = null;

function runComparison() {
    const weights = {
        space: parseFloat(document.getElementById('weight-space').value),
        accessibility: parseFloat(document.getElementById('weight-accessibility').value),
        stability: parseFloat(document.getElementById('weight-stability').value)
    };

    const modal = document.getElementById('comparison-modal');
    const content = document.getElementById('comparison-results-content');
    modal.style.display = 'flex';
    content.innerHTML = `
        <div style="text-align:center; padding: 20px;">
            <div style="font-size: 1.2rem; color: var(--accent-primary); margin-bottom: 15px;">Starting Algorithm Comparison...</div>
            <div class="dot active" style="margin:20px auto;"></div>
        </div>
    `;

    fetch(`${API_BASE_URL}/api/optimize/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            warehouse_id: currentWarehouseId,
            weights: weights
        })
    }).then(res => res.json())
        .then(data => {
            if (data.success) {
                // Start polling for status
                startComparisonPolling();
            } else {
                content.innerHTML = 'Error: ' + (data.error || JSON.stringify(data));
            }
        })
        .catch(err => {
            content.innerHTML = 'Error: ' + err;
        });
}

function startComparisonPolling() {
    const content = document.getElementById('comparison-results-content');

    comparisonInterval = setInterval(() => {
        fetch(`${API_BASE_URL}/api/optimize/compare/status`)
            .then(res => res.json())
            .then(status => {
                let html = `
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-size: 1.1rem; color: var(--accent-primary);">
                                Algorithm ${status.current_algorithm_index || 0}/${status.total_algorithms}
                            </span>
                            <span style="font-family: 'JetBrains Mono'; color: var(--success);">
                                ${Math.round(status.progress || 0)}%
                            </span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, var(--accent-primary), var(--success)); height: 100%; width: ${status.progress || 0}%; transition: width 0.3s;"></div>
                        </div>
                        <div style="margin-top: 10px; color: var(--text-muted); font-size: 0.9rem;">
                            ${status.message || 'Processing...'}
                        </div>
                    </div>
                `;

                // Show completed results as they come in
                if (status.results && Object.keys(status.results).length > 0) {
                    html += `<div style="margin-top: 15px; border-top: 1px solid #333; padding-top: 15px;">
                        <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 10px;">Completed Results:</div>`;

                    for (const [algo, result] of Object.entries(status.results)) {
                        const color = algo === 'GA' ? '#00f0ff' : algo === 'EO' ? '#7000ff' : algo.includes('GA-EO') ? '#ff0055' : '#ffd600';
                        const statusIcon = result.status === 'completed' ? '✓' : result.status === 'error' ? '✗' : '⋯';

                        html += `<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <span style="color: ${color}; font-weight: bold;">${statusIcon} ${algo}</span>
                            ${result.fitness ? `<span style="font-family: 'JetBrains Mono';">Fitness: ${result.fitness.toFixed(4)} | Time: ${result.time.toFixed(2)}s</span>` : `<span style="color: #ff6b8a;">${result.error || 'Running...'}</span>`}
                        </div>`;
                    }
                    html += '</div>';
                }

                content.innerHTML = html;

                // Check if done
                if (!status.running) {
                    clearInterval(comparisonInterval);
                    comparisonInterval = null;

                    // Show final results table
                    setTimeout(() => showFinalComparisonResults(status.results), 500);
                }
            })
            .catch(err => {
                console.error('Polling error:', err);
            });
    }, 500);
}

function showFinalComparisonResults(results) {
    const content = document.getElementById('comparison-results-content');

    if (!results || Object.keys(results).length === 0) {
        content.innerHTML = '<div style="color: var(--danger);">No results available.</div>';
        return;
    }

    let html = '<div style="margin-bottom: 15px; color: var(--success); font-size: 1.1rem;">✓ Comparison Complete!</div>';
    html += '<table style="width:100%; border-collapse:collapse; text-align:left;">';
    html += '<thead><tr style="border-bottom:1px solid #444; color:var(--accent-primary);">';
    html += '<th style="padding:10px;">Metric</th>';

    const algos = Object.keys(results);
    algos.forEach(algo => {
        const color = algo === 'GA' ? '#00f0ff' : algo === 'EO' ? '#7000ff' : algo.includes('GA-EO') ? '#ff0055' : '#ffd600';
        html += `<th style="padding:10px; color: ${color};">${algo}</th>`;
    });
    html += '</tr></thead><tbody>';

    const metrics = [
        { key: 'fitness', label: 'Total Fitness', higher: true },
        { key: 'time', label: 'Total Time (s)', higher: false },
        { key: 'time_to_best', label: 'Time to Best (s)', higher: false },
        { key: 'space_utilization', label: 'Space Util', higher: true },
        { key: 'accessibility', label: 'Accessibility', higher: true },
        { key: 'stability', label: 'Stability', higher: true }
    ];

    metrics.forEach(m => {
        // Find best value
        let bestVal = null;
        algos.forEach(algo => {
            const val = results[algo][m.key];
            if (typeof val === 'number') {
                if (bestVal === null) bestVal = val;
                else if (m.higher && val > bestVal) bestVal = val;
                else if (!m.higher && val < bestVal) bestVal = val;
            }
        });

        html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">`;
        html += `<td style="padding:10px; color:#fff;">${m.label}</td>`;

        algos.forEach(algo => {
            let val = results[algo][m.key];
            let displayVal = val;

            if (typeof val === 'number') {
                if (m.key === 'time' || m.key === 'time_to_best') displayVal = val.toFixed(3) + 's';
                else if (m.key === 'fitness' || m.key === 'accessibility') displayVal = val.toFixed(4);
                else displayVal = (val * 100).toFixed(1) + '%';
            }

            // Highlight winner
            const isBest = typeof val === 'number' && Math.abs(val - bestVal) < 0.0001;
            const color = isBest ? 'var(--success)' : '#8899A6';
            const weight = isBest ? 'bold' : 'normal';

            html += `<td style="padding:10px; color:${color}; font-weight:${weight};">${displayVal || '-'}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';

    // Add stop button replacement
    html += `<div style="margin-top: 20px; text-align: center;">
        <button class="btn" onclick="document.getElementById('comparison-modal').style.display='none'" 
                style="background: var(--accent-primary); padding: 10px 30px;">Close</button>
    </div>`;

    content.innerHTML = html;
}

// --- Benchmark Functions ---
let benchmarkInterval = null;

function runBenchmark() {
    if (!confirm('This will run each algorithm 20 times. This may take several minutes. Continue?')) {
        return;
    }

    const benchmarkBtn = document.getElementById('benchmark-btn');
    const statusDiv = document.getElementById('benchmark-status');

    if (benchmarkBtn) benchmarkBtn.disabled = true;
    if (statusDiv) statusDiv.style.display = 'block';

    const params = {
        warehouse_id: currentWarehouseId,
        runs: 20,
        generations: 50,  // Reduced for faster benchmarking
        iterations: 500,
        population_size: 30,
        weights: {
            space: parseFloat(document.getElementById('weight-space')?.value || 0.6),
            accessibility: parseFloat(document.getElementById('weight-accessibility')?.value || 0.3),
            stability: parseFloat(document.getElementById('weight-stability')?.value || 0.1)
        }
    };

    fetch(`${API_BASE_URL}/api/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                startBenchmarkPolling();
            } else {
                alert('Error starting benchmark: ' + data.error);
                if (benchmarkBtn) benchmarkBtn.disabled = false;
                if (statusDiv) statusDiv.style.display = 'none';
            }
        })
        .catch(err => {
            alert('Network error: ' + err);
            if (benchmarkBtn) benchmarkBtn.disabled = false;
            if (statusDiv) statusDiv.style.display = 'none';
        });
}

function startBenchmarkPolling() {
    benchmarkInterval = setInterval(() => {
        fetch(`${API_BASE_URL}/api/benchmark/status`)
            .then(res => res.json())
            .then(status => {
                const algoSpan = document.getElementById('benchmark-algo');
                const progressSpan = document.getElementById('benchmark-progress');
                const progressBar = document.getElementById('benchmark-bar');

                // Fix: 4 algorithms now (GA, EO, GA-EO, EO-GA)
                const runsPerAlgo = Math.ceil(status.total_runs / 4);
                if (algoSpan) algoSpan.textContent = `${status.current_algo || '-'} (Run ${status.current_run}/${runsPerAlgo})`;
                if (progressSpan) progressSpan.textContent = Math.round(status.progress) + '%';
                if (progressBar) progressBar.style.width = status.progress + '%';

                // Show real-time results as each algorithm completes
                const resultsDiv = document.getElementById('benchmark-results');
                if (resultsDiv && status.results && Object.keys(status.results).length > 0) {
                    let html = '<div style="margin-top: 10px; font-size: 0.8rem;">';
                    html += '<div style="color: var(--text-muted); margin-bottom: 5px;">Completed:</div>';
                    for (const [key, result] of Object.entries(status.results)) {
                        const color = key === 'GA' ? '#00f0ff' : key === 'EO' ? '#7000ff' : key === 'GA-EO' ? '#ff0055' : '#ffd600';
                        html += `<div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #333;">`;
                        html += `<span style="color: ${color}; font-weight: bold;">${key}</span>`;
                        html += `<span style="font-family: 'JetBrains Mono';">${result.avg_fitness.toFixed(4)} (${result.runs} runs)</span>`;
                        html += `</div>`;
                    }
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                }

                if (!status.running) {
                    clearInterval(benchmarkInterval);
                    benchmarkInterval = null;

                    const benchmarkBtn = document.getElementById('benchmark-btn');
                    const statusDiv = document.getElementById('benchmark-status');

                    if (benchmarkBtn) benchmarkBtn.disabled = false;
                    if (statusDiv) statusDiv.style.display = 'none';

                    // Refresh analytics to show new averaged results
                    loadAnalytics();
                    alert('Benchmark complete! Results have been saved.');
                }
            });
    }, 500);
}

function clearAlgoPerformance() {
    if (!confirm('This will delete all optimization results for this warehouse. Are you sure?')) {
        return;
    }

    fetch(`${API_BASE_URL}/api/metrics/algo-best/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ warehouse_id: currentWarehouseId })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Cleared ${data.deleted} optimization records.`);
                loadAnalytics(); // Refresh the table
            } else {
                alert('Error clearing performance: ' + data.error);
            }
        })
        .catch(err => {
            alert('Network error: ' + err);
        });
}

// Count actual 3D boxes rendered in the scene (not from items data)
function countRenderedBoxes() {
    const count = itemsGroup ? itemsGroup.children.length : 0;
    const mRendered = document.getElementById('metric-rendered');
    if (mRendered) {
        mRendered.textContent = count;
        mRendered.style.color = count > 0 ? '#00f0ff' : '#ff6b8a';
    }

    return count;
}

function exportAlgoPerformance() {
    fetch(`${API_BASE_URL}/api/metrics/algo-best?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            if (data.length === 0) {
                alert('No performance data to export.');
                return;
            }

            // CSV Header
            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Algorithm,Best Fitness,Space Util (%),Accessibility,Stability (%),Grouping,Time to Best (s),Execution Time (s),Achieved At\n";

            // CSV Body
            data.forEach(row => {
                const algo = row.algorithm;
                const bestFit = row.best_fitness ? row.best_fitness.toFixed(6) : "0";
                const spaceUtil = row.space_utilization ? (row.space_utilization * 100).toFixed(2) : "0";
                const access = row.accessibility ? row.accessibility.toFixed(6) : "0";
                const stab = row.stability ? (row.stability * 100).toFixed(2) : "0";
                const group = row.grouping ? row.grouping.toFixed(6) : "0";
                const timeBest = row.time_to_best ? row.time_to_best.toFixed(4) : "0";
                const execTime = row.execution_time ? row.execution_time.toFixed(4) : "0";
                const timestamp = row.timestamp ? new Date(row.timestamp).toLocaleString() : "-";

                csvContent += `"${algo}",${bestFit},${spaceUtil},${access},${stab},${group},${timeBest},${execTime},"${timestamp}"\n`;
            });

            // Trigger download
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `algorithm_performance_warehouse_${currentWarehouseId}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        })
        .catch(err => {
            console.error('Export failed:', err);
            alert('Failed to export data: ' + err);
        });
}