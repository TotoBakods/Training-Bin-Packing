// Global variables
let scene, camera, renderer, controls;
let warehouseGroup, itemsGroup, exclusionZonesGroup, pickerPathGroup, zoneLabelsGroup;
let coordinateLabels = [];
let warehouseConfig = {};
let allItemsData = {};
let comparisonResults = {}; // Cache for per-algorithm solutions AND metrics

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
    initResizer(); // Initialize UI resizer
    refreshViewModePicker(); // Populate previous results

    // Search input handler
    const searchInput = document.getElementById('item-search');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            console.log('Search input triggered:', searchInput.value);
            loadItemsList();
        });
    }

    // Delegated Item Deletion Handler (Robust for re-renders)
    const itemsListContainer = document.getElementById('items-list');
    if (itemsListContainer) {
        itemsListContainer.addEventListener('click', (e) => {
            const deleteBtn = e.target.closest('.delete-btn-item');
            if (deleteBtn) {
                const itemId = deleteBtn.getAttribute('data-id');
                if (itemId) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Delegated delete triggered for ID:', itemId);
                    deleteItem(itemId);
                }
            }
        });
    }
});

function initResizer() {
    const resizer = document.getElementById('resizer');
    const appContainer = document.getElementById('app-container');
    let isResizing = false;

    if (!resizer) return;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        resizer.classList.add('dragging');
        document.body.style.cursor = 'ew-resize';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        let newWidth = window.innerWidth - e.clientX;
        newWidth = Math.max(300, Math.min(newWidth, Math.floor(window.innerWidth * 0.5))); // Clamp
        
        appContainer.style.setProperty('--inspector-width', `${newWidth}px`);
        
        if (typeof onWindowResize === 'function') {
            onWindowResize(); // Force ThreeJS to adapt
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('dragging');
            document.body.style.cursor = '';
        }
    });
}

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
    zoneLabelsGroup = new THREE.Group();
    scene.add(warehouseGroup);
    scene.add(itemsGroup);
    scene.add(exclusionZonesGroup);
    scene.add(pickerPathGroup);
    scene.add(zoneLabelsGroup);

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

        // Categorize overflow: Z >= 2000 (Cascade/Physics fail) vs Z >= 1000 (Height exceed)
        const criticalFails = overflowItems.filter(i => i.z >= 2000).length;
        const heightExceeds = overflowItems.filter(i => i.z >= 1000 && i.z < 2000).length;

        // Build list of unplaced items
        const showCount = Math.min(overflowItems.length, 10);
        let itemsList = overflowItems.slice(0, showCount).map(item =>
            `<div style="padding: 2px 0; font-size: 0.75rem; color: #ccc;">
                • ${item.name || item.id} <span style="color:#888;">(${item.category || 'N/A'})</span>
                <span style="color:${item.z >= 2000 ? '#FF6B8A' : '#FFA500'}; font-size: 0.7rem;">
                    [${item.z >= 2000 ? 'Physics Fail' : 'Height Exceed'}]
                </span>
            </div>`
        ).join('');

        if (overflowItems.length > 10) {
            itemsList += `<div style="padding: 2px 0; font-size: 0.75rem; color: #888;">... and ${overflowItems.length - 10} more</div>`;
        }

        overflowIndicator.innerHTML = `
            <div style="cursor: pointer;" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none';">
                ⚠️ <strong>${overflowItems.length}</strong> items unplaced 
                <span style="font-size: 0.7rem; opacity: 0.8;">(${criticalFails} physics, ${heightExceeds} capacity)</span>
                <span style="font-size: 0.7rem; color: #888; float: right;">▼</span>
            </div>
            <div style="display: none; margin-top: 8px; max-height: 200px; overflow-y: auto; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 5px;">
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
    } else if (mode === 'priority') {
        // Priority 1 (Low) = Green, 2 = Yellow, 3 (High) = Red
        const p = item.priority || 1;
        if (p >= 3) return new THREE.Color(0xFF0055); // Red
        if (p === 2) return new THREE.Color(0xFFD600); // Yellow
        return new THREE.Color(0x00FF9D); // Green
    } else if (mode === 'stackable') {
        // Stackable = Cyan, Non-stackable = Muted Grey
        return item.stackable ? new THREE.Color(0x00F0FF) : new THREE.Color(0x444444);
    } else if (mode === 'displacement') {
        // Red (High Error/Displacement) -> Green (Low Error)
        const dx = item.x - (item.ml_x || item.x);
        const dy = item.y - (item.ml_y || item.y);
        const dz = item.z - (item.ml_z || item.z);
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const t = Math.min(1.0, dist / 2.0);
        const color = new THREE.Color().setHSL(0.33 - (t * 0.33), 1, 0.5); 
        return color;
    }
    return 0x888888;
}

// UI Event Handlers
function onColorModeChange(value) {
    const viewModeSelect = document.getElementById('view-mode-select');
    const currentView = viewModeSelect ? viewModeSelect.value : 'live';
    
    if (currentView !== 'live' && comparisonResults[currentView]) {
        // We are viewing a cached algorithm result
        const solution = comparisonResults[currentView].solution;
        const solutionItems = solution.map(sol => {
            const originalItem = allItemsData[sol.id];
            return originalItem ? { ...originalItem, ...sol } : null;
        }).filter(item => item !== null);
        renderItems(solutionItems);
    } else {
        // Normal live visualization update
        updateVisualization();
    }
}

function onViewModeChange(value) {
    const statusText = document.getElementById('status-text');
    
    if (value === 'live') {
        if (statusText) statusText.textContent = "LIVE VIEW";
        
        // Hide diagnostics when in Live View (not saved result)
        const diagPanel = document.getElementById('inference-diagnostics');
        if (diagPanel) diagPanel.style.display = 'none';
        
        updateVisualization();
        loadAnalytics();
        return;
    }

    if (comparisonResults[value]) {
        // Render from cache
        const res = comparisonResults[value];
        const solutionItems = res.solution.map(sol => {
            const originalItem = allItemsData[sol.id];
            return originalItem ? { ...originalItem, ...sol } : null;
        }).filter(item => item !== null);
        renderItems(solutionItems);
        
        // Update Metrics Dashboard
        updateMetricsDashboard(res.metrics, res.solution);
        
        if (statusText) statusText.textContent = `RESULT: ${value}`;
        
        // Update Optimization Tab Status
        const algoName = value.replace('ML - ', '').replace('_COMPARE', '');
        const statusAlgo = document.getElementById('status-algo');
        const statusElapsed = document.getElementById('status-elapsed');
        const statusGen = document.getElementById('status-gen');
        const statusPlaced = document.getElementById('status-placed');
        if (statusAlgo) statusAlgo.textContent = algoName;
        if (statusElapsed) statusElapsed.textContent = res.metrics.execution_time ? res.metrics.execution_time.toFixed(2) + 's' : '-';
        if (statusGen) statusGen.textContent = 'Completed';
        
        // Update Best Fitness
        const bestFit = document.getElementById('best-fitness');
        if (bestFit && res.metrics.fitness) bestFit.textContent = res.metrics.fitness.toFixed(4);
        
        // Calculate placed items (items with Z < 1000)
        if (statusPlaced && res.solution) {
            const placedCount = res.solution.filter(i => (i.z || 0) < 1000).length;
            statusPlaced.textContent = `${placedCount} / ${res.solution.length}`;
            
            const progressPercent = document.getElementById('progress-percent');
            const progressBarFill = document.getElementById('progress-bar-fill');
            if (progressPercent) progressPercent.textContent = '100%';
            if (progressBarFill) progressBarFill.style.width = '100%';
        }
    } else if (value.startsWith('ML - ')) {
        // Try fetching from API
        const algoName = value.endsWith('_COMPARE') ? value : value + '_COMPARE';
        fetch(`${API_BASE_URL}/api/metrics/solution?algorithm=${encodeURIComponent(algoName)}&warehouse_id=${currentWarehouseId}`)
            .then(res => res.json())
            .then(data => {
                if (data.success && data.solution) {
                    comparisonResults[value] = {
                        solution: data.solution,
                        metrics: data.metrics
                    };
                    onViewModeChange(value); // Recursive call now that it's cached
                } else {
                    console.error("Could not fetch result for", value);
                    onViewModeChange('live');
                }
            })
            .catch(err => {
                console.error("Error fetching result:", err);
                onViewModeChange('live');
            });
    }
}

function updateMetricsDashboard(m, solution = null) {
    if (!m) return;
    
    // Calculate missing metrics from solution if available
    let totalItems = m.total_items;
    let freeSpace = m.free_space_vol;
    
    if (solution && (!totalItems || !freeSpace)) {
        totalItems = solution.length;
        const fittingItems = solution.filter(i => (i.z || 0) < 1000);
        
        // Calculate volume if we have warehouseConfig
        if (warehouseConfig && warehouseConfig.length) {
            const whVol = warehouseConfig.length * warehouseConfig.width * warehouseConfig.height;
            let itemsVol = 0;
            fittingItems.forEach(i => {
                // We need original dimensions for volume calculation
                const orig = allItemsData[i.id];
                if (orig) {
                    itemsVol += orig.length * orig.width * orig.height;
                }
            });
            freeSpace = Math.max(0, whVol - itemsVol);
        }
    }

    // Standard Metrics
    const elements = {
        'metric-space': (m.space_utilization * 100).toFixed(1) + '%',
        'metric-runtime': m.execution_time ? m.execution_time.toFixed(2) + 's' : '0.00s',
        'metric-access': m.accessibility ? m.accessibility.toFixed(2) : '0.00',
        'metric-stability': (m.stability * 100).toFixed(1) + '%',
        'metric-count': totalItems || '-',
        'metric-free-space': freeSpace ? freeSpace.toFixed(2) + ' m³' : '-'
    };
    
    for (const [id, val] of Object.entries(elements)) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = val;
            el.style.color = '#FFD600'; // Highlight that this is a "saved" metric
        }
    }
    
    // Update fitness separately
    const bestFit = document.getElementById('best-fitness');
    if (bestFit && m.fitness) bestFit.textContent = m.fitness.toFixed(4);

    // Inference Diagnostics Panel
    const diagPanel = document.getElementById('inference-diagnostics');
    const inf = m.inference_metrics;
    
    if (inf && diagPanel) {
        diagPanel.style.display = 'block';
        
        const diagElements = {
            'diag-latency': (inf.total_latency_ms || 0).toFixed(1) + 'ms',
            'diag-displacement': (inf.avg_displacement || 0).toFixed(3) + 'm',
            'diag-efficiency': ((inf.volumetric_efficiency || 0) * 100).toFixed(1) + '%',
            'diag-ssr': (inf.ssr_pct || 0).toFixed(1) + '%',
            'diag-psr': (inf.psr_pct || 0).toFixed(1) + '%',
            'diag-placed': (inf.placed_count || 0)
        };
        
        for (const [id, val] of Object.entries(diagElements)) {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        }
    } else if (diagPanel) {
        diagPanel.style.display = 'none';
    }
}

function refreshViewModePicker() {
    const select = document.getElementById('view-mode-select');
    if (!select) return;

    const currentValue = select.value;
    select.innerHTML = '';

    // ALWAYS include Live View
    const liveOpt = document.createElement('option');
    liveOpt.value = 'live';
    liveOpt.textContent = 'LIVE / CURRENT STATE';
    select.appendChild(liveOpt);

    // Add Algorithm Results (Always available to fetch from DB)
    const standardAlgos = ['ML - GA', 'ML - EO', 'ML - Hybrid GA-EO', 'ML - Hybrid EO-GA'];
    
    const group = document.createElement('optgroup');
    group.label = 'ALGORITHM RESULTS (FETCH FROM DB)';
    standardAlgos.forEach(algo => {
        const opt = document.createElement('option');
        opt.value = algo;
        opt.textContent = 'Result: ' + algo.replace('ML - ', '');
        group.appendChild(opt);
    });
    select.appendChild(group);

    // Try to restore previous value or default
    if (Array.from(select.options).some(o => o.value === currentValue)) {
        select.value = currentValue;
    } else {
        select.value = 'live';
    }
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
    console.log("startOptimization() called");
    const algoSelect = document.getElementById('algorithm-select');
    if (!algoSelect) {
        console.error("algorithm-select not found");
        return;
    }
    const algo = algoSelect.value;

    const params = {
        warehouse_id: currentWarehouseId,
        weights: {
            space: parseFloat(document.getElementById('weight-space')?.value || 0.5),
            accessibility: parseFloat(document.getElementById('weight-accessibility')?.value || 0.4),
            stability: parseFloat(document.getElementById('weight-stability')?.value || 0.1)
        }
    };

    console.log("Optimization parameters:", params);

    const startBtn = document.getElementById('start-btn');
    if (startBtn) startBtn.disabled = true;

    // Load items before starting
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            console.log(`Loaded ${items.length} items for optimization`);
            if (items.length === 0) {
                alert("No items found to optimize. Please add items via CSV upload or use the SCRAMBLE button in Item Management.");
                if (startBtn) startBtn.disabled = false;
                return;
            }
            items.forEach(i => allItemsData[i.id] = i);
            startOptimizationRequest(algo, params);
        })
        .catch(err => {
            console.error("Optimization failed to start:", err);
            alert(`Failed to load items: ${err}`);
            if (startBtn) startBtn.disabled = false;
        });
}

let currentOptimizationType = null;
let optimization_state_start_time = 0;

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
                optimization_state_start_time = Date.now();
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
                    
                    // Update diagnostics if available in final status
                    if (status.inference_metrics) {
                        updateMetricsDashboard({
                            ...status,
                            inference_metrics: status.inference_metrics,
                            execution_time: (Date.now() - optimization_state_start_time) / 1000 // approx
                        });
                    }

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

                        // Cache the results for the view mode switcher
                        if (status.results) {
                            // The backend status return for compare usually includes the final solutions if we are polling
                            // But wait, the comparison_state in app.py has results but those are just metrics.
                            // However, during polling, 'best_solution' is updated for each algo.
                            // We need to make sure we capture the final state.
                            // For now, let's assume we want to call the analytics endpoint to get the best solutions if they aren't here.
                            // Actually, the app.py finalize_optimization saves to DB. 
                            // But we want to view them WITHOUT switching the DB state.
                            
                            // Let's check how the backend handles current state.
                            // If currentOptimizationType is 'compare', we should probably ask for the full results.
                            fetch(`${API_BASE_URL}/api/optimize/compare/status`)
                                .then(res => res.json())
                                .then(compStatus => {
                                    // Though compStatus.results doesn't have solutions, it likely has the best_solution 
                                    // of the LAST one run. 
                                    // To support switching between ALL, we'd need them stored.
                                    // I'll add a note that the backend might need an update if it doesn't return all solutions.
                                    refreshViewModePicker();
                                });
                        }
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

                // Update Inference Diagnostics
                if (status.inference_metrics) {
                    const diagPanel = document.getElementById('inference-diagnostics');
                    if (diagPanel) diagPanel.style.display = 'block';
                    
                    const meta = status.inference_metrics;
                    const dLat = document.getElementById('diag-latency');
                    const dDisp = document.getElementById('diag-displacement');
                    const dEff = document.getElementById('diag-efficiency');
                    const dPlac = document.getElementById('diag-placed');
                    
                    if (dLat) dLat.textContent = (meta.inference_latency_ms || 0).toFixed(1) + ' ms';
                    if (dDisp) dDisp.textContent = (meta.avg_displacement || 0).toFixed(3) + ' m';
                    if (dEff) dEff.textContent = ((meta.volumetric_efficiency || 0) * 100).toFixed(1) + '%';
                    if (dPlac) dPlac.textContent = meta.placed_count || '0';
                    
                    const dSsr = document.getElementById('diag-ssr');
                    const dPsr = document.getElementById('diag-psr');
                    if (dSsr) dSsr.textContent = (meta.ssr_pct || 0).toFixed(1) + '%';
                    if (dPsr) {
                        dPsr.textContent = (meta.psr_pct || 0).toFixed(1) + '%';
                        dPsr.style.color = (meta.psr_pct > 95) ? 'var(--success)' : 'var(--text-main)';
                    }
                } else {
                    const diagPanel = document.getElementById('inference-diagnostics');
                    if (diagPanel && !status.running) diagPanel.style.display = 'none';
                }

                if (status.best_solution && status.best_solution.length > 0) {
                    // Map solution coordinates back to full item data
                    const solutionItems = status.best_solution.map(sol => {
                        const originalItem = allItemsData[sol.id];
                        return originalItem ? { ...originalItem, ...sol } : null;
                    }).filter(item => item !== null);

                    if (solutionItems.length > 0) {
                        renderItems(solutionItems);
                        
                        if (currentOptimizationType === 'compare' && status.current_algorithm) {
                            comparisonResults[status.current_algorithm] = {
                                solution: JSON.parse(JSON.stringify(status.best_solution)),
                                metrics: {
                                    fitness: status.best_fitness || 0,
                                    space_utilization: status.space_utilization || 0,
                                    accessibility: status.accessibility || 0,
                                    stability: status.stability || 0,
                                    grouping: status.grouping || 0,
                                    execution_time: status.elapsed_time || 0
                                }
                            };
                        }
                    }
                }
            })
            .catch(err => console.error("Polling error:", err));
    }, 200); 
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

            if (mSpace) { mSpace.textContent = (data.space_utilization * 100).toFixed(1) + '%'; mSpace.style.color = ''; }
            if (mFreeSpace) { mFreeSpace.textContent = data.free_space_vol ? data.free_space_vol.toFixed(2) + ' m³' : '0.00 m³'; mFreeSpace.style.color = ''; }
            if (mRuntime) { mRuntime.textContent = data.execution_time ? data.execution_time.toFixed(2) + 's' : '0.00s'; mRuntime.style.color = ''; }
            if (mAccess) { mAccess.textContent = data.accessibility.toFixed(2); mAccess.style.color = ''; }
            if (mStab) { mStab.textContent = (data.stability * 100).toFixed(1) + '%'; mStab.style.color = ''; }
            if (mCount) { mCount.textContent = data.total_items; mCount.style.color = ''; }

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

// X-ray feature
let currentlyXrayedItemId = null;
window.xrayItem = function(itemId) {
    if (currentlyXrayedItemId === itemId) {
        currentlyXrayedItemId = null;
        updateVisualization();
        return;
    }
    currentlyXrayedItemId = itemId;
    itemsGroup.children.forEach(mesh => {
        if (mesh.userData && mesh.userData.id === itemId) {
            mesh.material.emissive.setHex(0x00A5FF);
            mesh.material.emissiveIntensity = 0.8;
            mesh.material.transparent = false;
            mesh.material.opacity = 1.0;
        } else {
            mesh.material.emissive.setHex(0x000000);
            mesh.material.transparent = true;
            mesh.material.opacity = 0.15;
        }
    });
};

// Item list
function loadItemsList() {
    console.log('loadItemsList called');
    const container = document.getElementById('items-list');
    const searchInput = document.getElementById('item-search');
    const query = searchInput ? searchInput.value.toLowerCase() : '';

    if (!query) container.innerHTML = '<div style="padding:10px; color:var(--text-muted);">Loading...</div>';

    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            let filteredItems = items;
            if (query) {
                filteredItems = items.filter(item => 
                    (item.name && item.name.toLowerCase().includes(query)) || 
                    (item.id && item.id.toLowerCase().includes(query)) ||
                    (item.category && item.category.toLowerCase().includes(query))
                );
            }

            container.innerHTML = '';
            
            if (filteredItems.length === 0) {
                container.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.8rem;">No items found ${query ? `matching "${query}"` : ''}</div>`;
                return;
            }

            const table = document.createElement('table');
            table.className = 'compact-table';
            table.style.marginTop = '8px';

            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            headerRow.style.color = 'var(--accent-primary)';
            headerRow.style.borderBottom = '1px solid #444';
            
            ['Name', 'Dim', 'Cat', 'F', 'Action'].forEach((text, idx) => {
                const th = document.createElement('th');
                th.textContent = text;
                const widths = ['35%', '25%', '15%', '10%', '15%'];
                th.style.width = widths[idx];
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            filteredItems.forEach(item => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                
                const tdName = document.createElement('td');
                tdName.textContent = item.name || item.id;
                tr.appendChild(tdName);

                const tdDim = document.createElement('td');
                tdDim.style.color = 'var(--accent-primary)';
                tdDim.textContent = `${item.width}x${item.height}x${item.length}`;
                tr.appendChild(tdDim);

                const tdCat = document.createElement('td');
                tdCat.textContent = item.category || 'N/A';
                tr.appendChild(tdCat);

                const tdFrag = document.createElement('td');
                tdFrag.textContent = item.fragility ? '⚠️' : '🛡️';
                tr.appendChild(tdFrag);

                const tdAction = document.createElement('td');
                tdAction.style.display = 'flex';
                tdAction.style.gap = '5px';
                
                const xrayBtn = document.createElement('button');
                xrayBtn.innerHTML = '👁️';
                xrayBtn.className = 'xray-btn-item'; 
                xrayBtn.title = 'Highlight (X-Ray)';
                xrayBtn.style.background = 'none';
                xrayBtn.style.border = 'none';
                xrayBtn.style.cursor = 'pointer';
                xrayBtn.onclick = () => window.xrayItem(item.id);
                tdAction.appendChild(xrayBtn);

                const btn = document.createElement('button');
                btn.innerHTML = '🗑️';
                btn.className = 'delete-btn-item'; 
                btn.setAttribute('data-id', item.id);
                btn.style.background = 'none';
                btn.style.border = 'none';
                btn.style.color = 'var(--danger, #ff4c4c)';
                btn.style.cursor = 'pointer';
                btn.style.padding = '0';
                btn.style.pointerEvents = 'auto'; // Ensure it's clickable
                btn.title = 'Delete Item';
                
                tdAction.appendChild(btn);
                tr.appendChild(tdAction);

                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            container.appendChild(table);
        })
        .catch(err => {
            console.error('loadItemsList error:', err);
            container.innerHTML = `<div style="color:var(--danger); padding:10px;">Error loading items: ${err.message}</div>`;
        });
}

function deleteItem(itemId) {
    console.log('deleteItem executing for ID:', itemId);
    if (!confirm('Are you sure you want to delete this item?')) return;
    
    // Safety: encode ID for URL
    const encodedId = encodeURIComponent(itemId);
    
    fetch(`${API_BASE_URL}/api/items/${encodedId}?warehouse_id=${currentWarehouseId}`, { method: 'DELETE' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                console.log('Successfully deleted ID:', itemId);
                loadItemsList();
                updateVisualization();
            } else {
                console.error('API Error during delete:', data.error);
                alert('Deletion failed: ' + data.error);
            }
        })
        .catch(err => {
            console.error('Network error during delete:', err);
            alert('A network error occurred while deleting the item.');
        });
}

function addNewItem() {
    const name = document.getElementById('new-item-name').value || 'New Item';
    const category = document.getElementById('new-item-category').value || 'General';
    const length = parseFloat(document.getElementById('new-item-length').value) || 1.0;
    const width = parseFloat(document.getElementById('new-item-width').value) || 1.0;
    const height = parseFloat(document.getElementById('new-item-height').value) || 1.0;
    const weight = parseFloat(document.getElementById('new-item-weight').value) || 10.0;
    const fragile = document.getElementById('new-item-fragile').checked;
    const stackable = document.getElementById('new-item-stackable').checked;

    const data = {
        id: 'ITEM-' + Date.now() + '-' + Math.floor(Math.random() * 1000),
        name,
        category,
        length,
        width,
        height,
        weight,
        fragility: fragile ? 1 : 0,
        stackable: stackable ? 1 : 0,
        access_freq: Math.random() * 0.5,
        can_rotate: !fragile ? 1 : 0,
        priority: 1,
        warehouse_id: currentWarehouseId
    };

    fetch(`${API_BASE_URL}/api/items`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        if (result.success) {
            loadItemsList();
            updateVisualization();
            rerunOptimization();
            // Clear inputs
            document.getElementById('new-item-name').value = '';
            document.getElementById('new-item-category').value = '';
            document.getElementById('new-item-length').value = '';
            document.getElementById('new-item-width').value = '';
            document.getElementById('new-item-height').value = '';
            document.getElementById('new-item-weight').value = '';
        } else {
            alert('Failed to add item: ' + result.error);
        }
    });
}

function rerunOptimization() {
    const viewModeSelect = document.getElementById('view-mode-select');
    const currentView = viewModeSelect ? viewModeSelect.value : 'live';

    let targetAlgorithm = 'ga'; // default

    if (currentView === 'live') {
        const algoSelect = document.getElementById('algorithm-select');
        if (algoSelect) targetAlgorithm = algoSelect.value;
    } else {
        // Parse custom result view name... ML - GA -> ga, ML - EO -> eo...
        if (currentView.includes('GA-EO')) targetAlgorithm = 'ga-eo';
        else if (currentView.includes('EO-GA')) targetAlgorithm = 'eo-ga';
        else if (currentView.includes('GA')) targetAlgorithm = 'ga';
        else if (currentView.includes('EO')) targetAlgorithm = 'eo';
    }

    // Set controls to match
    const algoSelect = document.getElementById('algorithm-select');
    if (algoSelect) {
        algoSelect.value = targetAlgorithm;
    }

    // Switch to live mode temporarily if not already so that startOptimization doesn't get confused
    if (viewModeSelect && viewModeSelect.value !== 'live') {
        viewModeSelect.value = 'live';
        onViewModeChange('live');
    }
    
    // Switch to controls view to show progress
    switchView('controls');
    
    // Start optimization
    startOptimization();
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
    while (zoneLabelsGroup.children.length > 0) {
        disposeObject(zoneLabelsGroup.children[0]);
        zoneLabelsGroup.remove(zoneLabelsGroup.children[0]);
    }

    const showLabels = document.getElementById('show-zone-names') ? document.getElementById('show-zone-names').checked : true;

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
        exclusionZonesGroup.add(mesh);

        // Add label if it's an allocation zone and labels are enabled
        if (isAlloc && showLabels && zone.name) {
            const labelPos = new THREE.Vector3(
                (zone.x1 + width / 2) - whLength / 2,
                z2 + 0.5, // Slightly above the top of the zone
                (zone.y1 + depth / 2) - whWidth / 2
            );
            createZoneLabel(zone.name, labelPos);
        }

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
            name: "Long Hall (Compact)", length: 20, width: 4, height: 6, levels: 2, grid_size: 1,
            door_x: 0, door_y: 2, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(() => {
            clearZones().then(() => {
                const zones = [];
                const racks = [
                    { name: 'Left Wall', x1: 1, y1: 0.5, x2: 19, y2: 1.5 },
                    { name: 'Right Wall', x1: 1, y1: 2.5, x2: 19, y2: 3.5 }
                ];
                racks.forEach(r => {
                    zones.push({ ...r, name: `${r.name} (Lower)`, z1: 0, z2: 3, zone_type: 'allocation', metadata: { levels: 1 } });
                    zones.push({ ...r, name: `${r.name} (Upper)`, z1: 3, z2: 6, zone_type: 'allocation', metadata: { levels: 1 } });
                });
                const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                Promise.all(promises).then(() => { loadWarehouseConfig(); alert('Applied: Compact Long Hall (20x4m, 4 Zones, 216m³ capacity)'); });
            });
        });
    } else if (name === 'large-zoned') {
        const newConfig = {
            name: "Large Zoned Warehouse", length: 60, width: 40, height: 10, levels: 3, grid_size: 2,
            door_x: 30, door_y: 0, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(() => {
            clearZones().then(() => {
                fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
                .then(res => res.json())
                .then(items => {
                    const categories = [...new Set(items.map(i => i.category || 'General'))];
                    const zones = [];
                    const cols = Math.ceil(Math.sqrt(categories.length)) || 1;
                    const rows = Math.ceil(categories.length / cols) || 1;
                    
                    const cw = 60 / cols;
                    const cd = 40 / rows;
                    
                    categories.forEach((cat, idx) => {
                        const r = Math.floor(idx / cols);
                        const c = idx % cols;
                        zones.push({
                            name: `${cat} Zone`,
                            x1: c * cw + 2, y1: r * cd + 2,
                            x2: (c + 1) * cw - 2, y2: (r + 1) * cd - 2,
                            z1: 0, z2: 10,
                            zone_type: 'allocation',
                            metadata: { levels: 3 }
                        });
                    });
                    
                    if (zones.length === 0) {
                        loadWarehouseConfig();
                        alert('Applied: Large Zoned Warehouse. No items/categories found to create zones.');
                        return;
                    }
                    
                    const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                    Promise.all(promises).then(() => { loadWarehouseConfig(); alert(`Applied: Large Zoned (${categories.length} Category Zones created)`); });
                });
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
            name: "L-Shape (Compact)", length: 10, width: 10, height: 6, levels: 2, grid_size: 1,
            door_x: 5, door_y: 0, id: currentWarehouseId
        };
        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newConfig)
        }).then(res => res.json()).then(d => {
            if (d.success) {
                clearZones().then(() => {
                    const zones = [];
                    // Compact Cutout
                    zones.push({
                        name: "L-Shape Cutout", x1: 5, y1: 5, x2: 10, y2: 10,
                        z1: 0, z2: 6, zone_type: 'exclusion', metadata: {}
                    });
                    
                    const racks = [
                        { name: 'Rack Arm A', x1: 1, y1: 1, x2: 4, y2: 7 },
                        { name: 'Rack Arm B', x1: 4, y1: 1, x2: 10, y2: 4 }
                    ];
                    racks.forEach(r => {
                        zones.push({ ...r, name: `${r.name} (Lower)`, z1: 0, z2: 3, zone_type: 'allocation', metadata: { levels: 1 } });
                        zones.push({ ...r, name: `${r.name} (Upper)`, z1: 3, z2: 6, zone_type: 'allocation', metadata: { levels: 1 } });
                    });

                    const promises = zones.map(z => fetch(`${API_BASE_URL}/api/warehouse/zones`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId }) }));
                    Promise.all(promises).then(() => { loadWarehouseConfig(); alert('Applied: Compact L-Shape (10x10m, 5 Zones, 216m³ capacity)'); });
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
                loadItemsList();
                updateVisualization();
                alert('All items deleted successfully.');
            } else {
                alert('Error deleting items: ' + data.error);
            }
        })
        .catch(error => console.error('Error:', error));
}

function clearAlgoPerformance() {
    if (!confirm('Clear all algorithm performance history for this warehouse? This will reset the comparison charts.')) {
        return;
    }

    fetch(`${API_BASE_URL}/api/metrics/all?warehouse_id=${currentWarehouseId}`, {
        method: 'DELETE'
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                // Refresh the analytics view
                loadAnalytics();
                // If we have a view mode picker (for comparing), refresh it to clear the options
                if (typeof refreshViewModePicker === 'function') {
                    refreshViewModePicker();
                }
                alert('Performance history cleared.');
            } else {
                alert('Failed to clear performance: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(err => {
            console.error('Clear performance error:', err);
            alert('Error clearing performance history.');
        });
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

function uploadCSV(isAppend = false) {
    const input = isAppend ? document.getElementById('file-upload-append') : document.getElementById('file-upload');
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    const appendParam = isAppend ? '&append=true' : '';

    fetch(`${API_BASE_URL}/api/upload-csv?warehouse_id=${currentWarehouseId}${appendParam}`, {
        method: 'POST',
        body: formData
    }).then(res => res.json())
        .then(d => {
            if (d.success) {
                alert('Uploaded Successfully');
                updateVisualization();
                if (document.getElementById('view-items').style.display === 'block') loadItemsList();
                rerunOptimization();
            } else {
                alert('Error: ' + d.error);
            }
            input.value = ''; // Reset input so same file can trigger change
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
/**
 * 3D Visualization Helper: Creates high-performance canvas labels for allocation zones
 */
function createZoneLabel(text, pos) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, 256, 64);
    
    // Professional rounded rect handling with backward compatibility
    ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(0, 0, 256, 64, 12);
    } else {
        // Fallback for older browsers
        ctx.rect(0, 0, 256, 64);
    }
    ctx.fill();
    ctx.strokeStyle = '#00F0FF';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = '#00F0FF';
    ctx.font = 'bold 30px "Outfit", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0, 240, 255, 0.5)';
    ctx.shadowBlur = 10;
    ctx.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.anisotropy = 16;
    const spriteMat = new THREE.SpriteMaterial({ 
        map: texture,
        transparent: true,
        depthTest: false,
        sizeAttenuation: true
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.copy(pos);
    sprite.scale.set(4, 1, 1);
    sprite.renderOrder = 100; // Ensure names are on top

    if (zoneLabelsGroup) zoneLabelsGroup.add(sprite);
}

/**
 * Toggles visibility of all 3D zone labels in the warehouse
 */
function toggleZoneLabels() {
    const checkbox = document.getElementById('show-zone-names');
    if (checkbox && zoneLabelsGroup) {
        zoneLabelsGroup.visible = checkbox.checked;
        if (checkbox.checked) {
            loadZones(); // Force re-render to ensure labels are fresh
        }
    }
}
