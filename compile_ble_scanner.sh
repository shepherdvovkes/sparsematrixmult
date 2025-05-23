#!/bin/bash

# BLE Scanner macOS Compilation Script for M1 Mac
# This script sets up and compiles the BLE Scanner application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="BLEScanner"
BUNDLE_ID="com.yourcompany.blescanner"
BUILD_DIR="build"
DERIVED_DATA_PATH="$BUILD_DIR/DerivedData"
ARCHIVE_PATH="$BUILD_DIR/$APP_NAME.xcarchive"
EXPORT_PATH="$BUILD_DIR/Export"

echo -e "${BLUE}🔷 BLE Scanner M1 Compilation Script${NC}"
echo "============================================="

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: This script is optimized for Apple Silicon (M1/M2) Macs${NC}"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check if full Xcode is installed (not just command line tools)
DEVELOPER_DIR=$(xcode-select -p 2>/dev/null || echo "")
XCODE_FULL_INSTALL=false

if [[ "$DEVELOPER_DIR" == *"Xcode.app"* ]]; then
    XCODE_FULL_INSTALL=true
    echo -e "${GREEN}✅ Full Xcode installation detected${NC}"
elif command_exists xcodebuild 2>/dev/null && xcodebuild -version >/dev/null 2>&1; then
    XCODE_FULL_INSTALL=true
    echo -e "${GREEN}✅ Xcode tools available${NC}"
else
    echo -e "${YELLOW}⚠️  Full Xcode not detected, using Command Line Tools only${NC}"
fi

if ! command_exists xcrun; then
    echo -e "${RED}❌ Error: xcrun not found${NC}"
    echo "Please install Xcode Command Line Tools with: xcode-select --install"
    exit 1
fi

if ! command_exists swift; then
    echo -e "${RED}❌ Error: Swift compiler not found${NC}"
    echo "Please install Xcode Command Line Tools"
    exit 1
fi

echo -e "${GREEN}✅ Swift compiler found${NC}"

# Check Xcode version if available
if $XCODE_FULL_INSTALL; then
    XCODE_VERSION=$(xcodebuild -version 2>/dev/null | head -n 1 | awk '{print $2}' || echo "Unknown")
    echo -e "${GREEN}✅ Xcode version: $XCODE_VERSION${NC}"
fi

# Check if Swift file exists in current directory
SWIFT_FILE=""
if [ -f "ble_scanner_macos.swift" ]; then
    SWIFT_FILE="ble_scanner_macos.swift"
    echo -e "${GREEN}✅ Found existing Swift file: $SWIFT_FILE${NC}"
elif [ -f "main.swift" ]; then
    SWIFT_FILE="main.swift"
    echo -e "${GREEN}✅ Found existing Swift file: $SWIFT_FILE${NC}"
elif [ -f "$APP_NAME.swift" ]; then
    SWIFT_FILE="$APP_NAME.swift"
    echo -e "${GREEN}✅ Found existing Swift file: $SWIFT_FILE${NC}"
else
    echo -e "${YELLOW}⚠️  No Swift file found, creating basic structure...${NC}"
fi

# Create or update project structure
if [ ! -d "$APP_NAME" ] || [ -n "$SWIFT_FILE" ]; then
    echo -e "${BLUE}📁 Setting up compilation environment...${NC}"
    
    # If we have an existing Swift file, use it directly for compilation
    if [ -n "$SWIFT_FILE" ]; then
        echo -e "${BLUE}📝 Using existing Swift file: $SWIFT_FILE${NC}"
    else
        # Create basic structure only if no Swift file exists
        mkdir -p "$APP_NAME"
        cd "$APP_NAME"
        
        # Create basic Swift file
        cat > "main.swift" << 'EOF'
import SwiftUI
import CoreBluetooth
import Combine

@main
struct BLEScannerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        Text("BLE Scanner - Replace with full code")
            .padding()
    }
}
EOF
        
        SWIFT_FILE="main.swift"
        cd ..
    fi
    
    echo -e "${GREEN}✅ Compilation environment ready${NC}"
else
    echo -e "${GREEN}✅ Project directory exists${NC}"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Method 1: Try Xcode project compilation (if .xcodeproj exists and full Xcode available)
if [ -d "$APP_NAME.xcodeproj" ] && $XCODE_FULL_INSTALL; then
    echo -e "${BLUE}🔨 Building with Xcode project...${NC}"
    
    # Clean previous builds
    echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
    xcodebuild clean \
        -project "$APP_NAME.xcodeproj" \
        -scheme "$APP_NAME" \
        -derivedDataPath "$DERIVED_DATA_PATH"
    
    # Build for M1 (arm64) architecture
    echo -e "${BLUE}🔨 Building for Apple Silicon...${NC}"
    xcodebuild build \
        -project "$APP_NAME.xcodeproj" \
        -scheme "$APP_NAME" \
        -configuration Release \
        -arch arm64 \
        -derivedDataPath "$DERIVED_DATA_PATH" \
        ONLY_ACTIVE_ARCH=NO \
        ARCHS="arm64" \
        VALID_ARCHS="arm64"
    
    # Archive the app
    echo -e "${BLUE}📦 Archiving application...${NC}"
    xcodebuild archive \
        -project "$APP_NAME.xcodeproj" \
        -scheme "$APP_NAME" \
        -configuration Release \
        -arch arm64 \
        -archivePath "$ARCHIVE_PATH" \
        -derivedDataPath "$DERIVED_DATA_PATH"
    
    # Export the app
    echo -e "${BLUE}📤 Exporting application...${NC}"
    xcodebuild -exportArchive \
        -archivePath "$ARCHIVE_PATH" \
        -exportPath "$EXPORT_PATH" \
        -exportOptionsPlist ExportOptions.plist 2>/dev/null || {
        
        # Create ExportOptions.plist if it doesn't exist
        cat > ExportOptions.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>mac-application</string>
    <key>destination</key>
    <string>export</string>
</dict>
</plist>
EOF
        
        xcodebuild -exportArchive \
            -archivePath "$ARCHIVE_PATH" \
            -exportPath "$EXPORT_PATH" \
            -exportOptionsPlist ExportOptions.plist
    }

# Method 2: Direct Swift compilation (preferred method when Swift file exists)
elif [ -n "$SWIFT_FILE" ]; then
    echo -e "${BLUE}🔨 Building with direct Swift compilation using $SWIFT_FILE...${NC}"
    
    # Get macOS SDK path
    MACOS_SDK=$(xcrun --show-sdk-path --sdk macosx 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    
    echo -e "${BLUE}🔧 Using SDK: $MACOS_SDK${NC}"
    
    # Compile the Swift file
    echo -e "${BLUE}🔨 Compiling $SWIFT_FILE for Apple Silicon (M1)...${NC}"
    
    # Clean any previous builds
    rm -f "$BUILD_DIR/$APP_NAME"
    
    # Try optimized compilation first
    if swiftc "$SWIFT_FILE" \
        -target arm64-apple-macos12.0 \
        -sdk "$MACOS_SDK" \
        -o "$BUILD_DIR/$APP_NAME" \
        -framework SwiftUI \
        -framework CoreBluetooth \
        -framework Foundation \
        -framework AppKit \
        -framework Combine \
        -Xlinker -rpath \
        -Xlinker /usr/lib/swift \
        -O 2>/dev/null; then
        echo -e "${GREEN}✅ Optimized compilation successful${NC}"
    
    # Try without optimization
    elif swiftc "$SWIFT_FILE" \
        -o "$BUILD_DIR/$APP_NAME" \
        -sdk "$MACOS_SDK" \
        -target arm64-apple-macos12.0 \
        -framework SwiftUI \
        -framework CoreBluetooth \
        -framework Foundation \
        -framework AppKit \
        -framework Combine 2>/dev/null; then
        echo -e "${GREEN}✅ Standard compilation successful${NC}"
    
    # Try with minimal flags
    elif swiftc "$SWIFT_FILE" \
        -o "$BUILD_DIR/$APP_NAME" \
        -target arm64-apple-macos12.0 \
        -framework SwiftUI \
        -framework CoreBluetooth 2>/dev/null; then
        echo -e "${GREEN}✅ Minimal compilation successful${NC}"
    
    # Final attempt with verbose output
    else
        echo -e "${RED}❌ Compilation failed. Trying with verbose output...${NC}"
        swiftc "$SWIFT_FILE" \
            -o "$BUILD_DIR/$APP_NAME" \
            -target arm64-apple-macos12.0 \
            -sdk "$MACOS_SDK" \
            -framework SwiftUI \
            -framework CoreBluetooth \
            -framework Foundation \
            -framework AppKit \
            -framework Combine \
            -v || {
            
            echo -e "${RED}❌ All compilation attempts failed${NC}"
            echo -e "${YELLOW}💡 Possible issues:${NC}"
            echo -e "${YELLOW}   - Missing Xcode Command Line Tools: xcode-select --install${NC}"
            echo -e "${YELLOW}   - Syntax errors in Swift file${NC}"
            echo -e "${YELLOW}   - Missing frameworks or dependencies${NC}"
            exit 1
        }
    fi
    
    # Create proper app bundle
    echo -e "${BLUE}📦 Creating app bundle...${NC}"
    APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
    mkdir -p "$APP_BUNDLE/Contents/MacOS"
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    
    # Move executable to app bundle
    cp "$BUILD_DIR/$APP_NAME" "$APP_BUNDLE/Contents/MacOS/"
    
    # Create comprehensive Info.plist for the app bundle
    cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>BLE Scanner</string>
    <key>CFBundleDisplayName</key>
    <string>BLE Scanner</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSBluetoothAlwaysUsageDescription</key>
    <string>This app needs Bluetooth access to scan for BLE devices and monitor packets.</string>
    <key>NSBluetoothPeripheralUsageDescription</key>
    <string>This app needs Bluetooth access to scan for BLE devices and monitor packets.</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

# Method 3: Swift Package Manager (if Package.swift exists)
elif [ -f "$APP_NAME/Package.swift" ]; then
    echo -e "${BLUE}🔨 Building with Swift Package Manager...${NC}"
    
    cd "$APP_NAME"
    
    # Build for M1
    echo -e "${BLUE}🔨 Building for Apple Silicon...${NC}"
    swift build -c release --arch arm64
    
    # Create app bundle structure
    echo -e "${BLUE}📦 Creating app bundle...${NC}"
    APP_BUNDLE="../$BUILD_DIR/$APP_NAME.app"
    mkdir -p "$APP_BUNDLE/Contents/MacOS"
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    
    # Copy executable
    cp ".build/release/$APP_NAME" "$APP_BUNDLE/Contents/MacOS/"
    
    # Create Info.plist for app bundle
    cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSBluetoothAlwaysUsageDescription</key>
    <string>This app needs Bluetooth access to scan for BLE devices and monitor packets.</string>
    <key>NSBluetoothPeripheralUsageDescription</key>
    <string>This app needs Bluetooth access to scan for BLE devices and monitor packets.</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
</dict>
</plist>
EOF
    
    cd ..

# Method 4: Fallback compilation
else
    echo -e "${YELLOW}⚠️  No suitable build method found${NC}"
    echo -e "${BLUE}💡 Creating minimal Swift file for compilation...${NC}"
    
    # Create a minimal Swift file as fallback
    cat > "fallback_main.swift" << 'EOF'
import SwiftUI

@main
struct BLEScannerApp: App {
    var body: some Scene {
        WindowGroup {
            Text("BLE Scanner - Please replace with full implementation")
                .padding()
        }
    }
}
EOF
    
    # Compile the fallback
    MACOS_SDK=$(xcrun --show-sdk-path --sdk macosx 2>/dev/null)
    swiftc fallback_main.swift \
        -target arm64-apple-macos12.0 \
        -sdk "$MACOS_SDK" \
        -o "$BUILD_DIR/$APP_NAME" \
        -framework SwiftUI \
        -framework Foundation \
        -framework AppKit
fi
    
    # First, let's create a proper Swift file with the full BLE Scanner code
    echo -e "${YELLOW}📝 Note: Creating basic Swift file. Replace with full BLE Scanner code for complete functionality.${NC}"
    
    # Create a standalone Swift file
    cat > "main.swift" << 'EOF'
import SwiftUI
import CoreBluetooth

@main
struct BLEScannerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowResizability(.contentSize)
    }
}

struct ContentView: View {
    @StateObject private var bleManager = SimpleBLEManager()
    
    var body: some View {
        VStack {
            Text("BLE Scanner")
                .font(.largeTitle)
                .padding()
            
            Text("Bluetooth Status: \(bleManager.statusText)")
                .foregroundColor(bleManager.isEnabled ? .green : .red)
                .padding()
            
            Button(bleManager.isScanning ? "Stop Scanning" : "Start Scanning") {
                if bleManager.isScanning {
                    bleManager.stopScanning()
                } else {
                    bleManager.startScanning()
                }
            }
            .disabled(!bleManager.isEnabled)
            .padding()
            
            List(bleManager.devices, id: \.identifier) { device in
                VStack(alignment: .leading) {
                    Text(device.name ?? "Unknown Device")
                        .font(.headline)
                    Text("RSSI: \(device.rssi) dBm")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(device.identifier.uuidString)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 2)
            }
        }
        .padding()
    }
}

class SimpleBLEManager: NSObject, ObservableObject, CBCentralManagerDelegate {
    private var centralManager: CBCentralManager!
    
    @Published var devices: [CBPeripheral] = []
    @Published var isScanning = false
    @Published var isEnabled = false
    @Published var statusText = "Unknown"
    
    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    func startScanning() {
        guard isEnabled else { return }
        centralManager.scanForPeripherals(withServices: nil, options: [CBCentralManagerScanOptionAllowDuplicatesKey: false])
        isScanning = true
    }
    
    func stopScanning() {
        centralManager.stopScan()
        isScanning = false
    }
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        DispatchQueue.main.async {
            switch central.state {
            case .poweredOn:
                self.isEnabled = true
                self.statusText = "Powered On"
            case .poweredOff:
                self.isEnabled = false
                self.statusText = "Powered Off"
            case .unauthorized:
                self.isEnabled = false
                self.statusText = "Unauthorized"
            case .unsupported:
                self.isEnabled = false
                self.statusText = "Unsupported"
            default:
                self.isEnabled = false
                self.statusText = "Unknown"
            }
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        DispatchQueue.main.async {
            if !self.devices.contains(where: { $0.identifier == peripheral.identifier }) {
                self.devices.append(peripheral)
            }
        }
    }
}
EOF
    
    # Get macOS SDK path
    MACOS_SDK=$(xcrun --show-sdk-path --sdk macosx 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    
    echo -e "${BLUE}🔧 Using SDK: $MACOS_SDK${NC}"
    
    # Compile the Swift file
    echo -e "${BLUE}🔨 Compiling Swift code...${NC}"
    swiftc main.swift \
        -target arm64-apple-macos12.0 \
        -sdk "$MACOS_SDK" \
        -o "$BUILD_DIR/$APP_NAME" \
        -framework SwiftUI \
        -framework CoreBluetooth \
        -framework Foundation \
        -framework AppKit \
        -Xlinker -rpath \
        -Xlinker /usr/lib/swift \
        2>/dev/null || {
        
        echo -e "${YELLOW}⚠️  Trying alternative compilation...${NC}"
        # Alternative compilation method
        swiftc main.swift \
            -o "$BUILD_DIR/$APP_NAME" \
            -sdk "$MACOS_SDK" \
            -target arm64-apple-macos12.0 || {
            
            echo -e "${RED}❌ Direct compilation failed${NC}"
            echo -e "${YELLOW}💡 Try installing full Xcode from the App Store for better compatibility${NC}"
            exit 1
        }
    }
    
    # Create proper app bundle
    echo -e "${BLUE}📦 Creating app bundle...${NC}"
    APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
    mkdir -p "$APP_BUNDLE/Contents/MacOS"
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    
    # Move executable to app bundle
    cp "$BUILD_DIR/$APP_NAME" "$APP_BUNDLE/Contents/MacOS/"
    
    # Create comprehensive Info.plist for the app bundle
    cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>BLE Scanner</string>
    <key>CFBundleDisplayName</key>
    <string>BLE Scanner</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSBluetoothAlwaysUsageDescription</key>
    <string>This app needs Bluetooth access to scan for BLE devices and monitor packets.</string>
    <key>NSBluetoothPeripheralUsageDescription</key>
    <string>This app needs Bluetooth access to scan for BLE devices and monitor packets.</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF
fi

# Verify the build
echo -e "${BLUE}🔍 Verifying build...${NC}"

if [ -f "$BUILD_DIR/$APP_NAME.app/Contents/MacOS/$APP_NAME" ]; then
    BINARY_PATH="$BUILD_DIR/$APP_NAME.app/Contents/MacOS/$APP_NAME"
elif [ -f "$EXPORT_PATH/$APP_NAME.app/Contents/MacOS/$APP_NAME" ]; then
    BINARY_PATH="$EXPORT_PATH/$APP_NAME.app/Contents/MacOS/$APP_NAME"
elif [ -f "$BUILD_DIR/$APP_NAME" ]; then
    BINARY_PATH="$BUILD_DIR/$APP_NAME"
else
    echo -e "${RED}❌ Build failed - executable not found${NC}"
    exit 1
fi

# Check architecture
ARCH=$(file "$BINARY_PATH" | grep -o "arm64\|x86_64")
echo -e "${GREEN}✅ Binary architecture: $ARCH${NC}"

if [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}✅ Successfully built for Apple Silicon (M1/M2)${NC}"
else
    echo -e "${YELLOW}⚠️  Built for $ARCH architecture${NC}"
fi

# Sign the app (development signing)
echo -e "${BLUE}✍️  Code signing...${NC}"
if [ -d "$BUILD_DIR/$APP_NAME.app" ]; then
    codesign --force --deep --sign - "$BUILD_DIR/$APP_NAME.app" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  Code signing skipped (no signing identity)${NC}"
fi

# Final success message
echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo "============================================="

if [ -d "$BUILD_DIR/$APP_NAME.app" ]; then
    echo -e "${GREEN}📱 App bundle: $BUILD_DIR/$APP_NAME.app${NC}"
    echo -e "${BLUE}💡 To run: open $BUILD_DIR/$APP_NAME.app${NC}"
elif [ -d "$EXPORT_PATH/$APP_NAME.app" ]; then
    echo -e "${GREEN}📱 App bundle: $EXPORT_PATH/$APP_NAME.app${NC}"
    echo -e "${BLUE}💡 To run: open $EXPORT_PATH/$APP_NAME.app${NC}"
else
    echo -e "${GREEN}🔧 Executable: $BINARY_PATH${NC}"
    echo -e "${BLUE}💡 To run: $BINARY_PATH${NC}"
fi

echo -e "${BLUE}💡 Note: The app may require Bluetooth permissions when first run${NC}"

# Optional: Open the build directory
read -p "Open build directory in Finder? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open "$BUILD_DIR"
fi