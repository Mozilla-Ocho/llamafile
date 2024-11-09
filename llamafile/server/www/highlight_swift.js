// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const SWIFT_KEYWORDS = new Set([
  'Any',
  'Protocol',
  'Self',
  'Sendable',
  'Type',
  '_BridgeObject',
  '_Class',
  '_NativeClass',
  '_NativeRefCountedObject',
  '_PackageDescription',
  '_RefCountedObject',
  '_Trivial',
  '_TrivialAtMost',
  '_TrivialStride',
  '_UnknownLayout',
  '__consuming',
  '__owned',
  '__setter_access',
  '__shared',
  '_alignment',
  '_backDeploy',
  '_borrow',
  '_borrowing',
  '_cdecl',
  '_compilerInitialized',
  '_const',
  '_consuming',
  '_documentation',
  '_dynamicReplacement',
  '_effects',
  '_expose',
  '_forward',
  '_implements',
  '_linear',
  '_local',
  '_modify',
  '_move',
  '_mutating',
  '_noMetadata',
  '_nonSendable',
  '_objcImplementation',
  '_objcRuntimeName',
  '_opaqueReturnTypeOf',
  '_optimize',
  '_originallyDefinedIn',
  '_private',
  '_projectedValueProperty',
  '_read',
  '_semantics',
  '_specialize',
  '_spi',
  '_spi_available',
  '_swift_native_objc_runtime_base',
  '_typeEraser',
  '_unavailableFromAsync',
  '_underlyingVersion',
  '_version',
  'accesses',
  'actor',
  'addressWithNativeOwner',
  'addressWithOwner',
  'any',
  'as',
  'assignment',
  'associatedtype',
  'associativity',
  'async',
  'attached',
  'autoclosure',
  'availability',
  'available',
  'await',
  'backDeployed',
  'before',
  'block',
  'borrowing',
  'break',
  'cType',
  'canImport',
  'case',
  'catch',
  'class',
  'compiler',
  'consume',
  'consuming',
  'continue',
  'convenience',
  'convention',
  'copy',
  'default',
  'defer',
  'deinit',
  'dependsOn',
  'deprecated',
  'derivative',
  'didSet',
  'differentiable',
  'discard',
  'distributed',
  'do',
  'dynamic',
  'each',
  'else',
  'enum',
  'escaping',
  'exclusivity',
  'exported',
  'extension',
  'fallthrough',
  'file',
  'fileprivate',
  'final',
  'for',
  'forward',
  'freestanding',
  'func',
  'get',
  'guard',
  'higherThan',
  'if',
  'import',
  'in',
  'indirect',
  'infix',
  'init',
  'initializes',
  'inline',
  'inout',
  'internal',
  'introduced',
  'is',
  'isolated',
  'kind',
  'lazy',
  'left',
  'let',
  'line',
  'linear',
  'lowerThan',
  'macro',
  'message',
  'metadata',
  'modify',
  'module',
  'mutableAddressWithNativeOwner',
  'mutableAddressWithOwner',
  'mutating',
  'noDerivative',
  'noasync',
  'noescape',
  'none',
  'nonisolated',
  'nonmutating',
  'objc',
  'obsoleted',
  'of',
  'open',
  'operator',
  'optional',
  'override',
  'package',
  'postfix',
  'precedencegroup',
  'preconcurrency',
  'prefix',
  'private',
  'protocol',
  'public',
  'read',
  'reasync',
  'renamed',
  'repeat',
  'required',
  'rethrows',
  'retroactive',
  'return',
  'reverse',
  'right',
  'safe',
  'scoped',
  'self',
  'sending',
  'set',
  'some',
  'sourceFile',
  'spi',
  'spiModule',
  'static',
  'struct',
  'subscript',
  'super',
  'swift',
  'switch',
  'target',
  'then',
  'throw',
  'throws',
  'transpose',
  'try',
  'typealias',
  'unavailable',
  'unchecked',
  'unowned',
  'unsafe',
  'unsafeAddress',
  'unsafeMutableAddress',
  'var',
  'visibility',
  'weak',
  'where',
  'while',
  'willSet',
  'witness_method',
  'wrt',
  'yield',
]);

const SWIFT_BUILTINS = new Set([
  'abs',
  'assert',
  'assertionFailure',
  'debugPrint',
  'dump',
  'fatalError',
  'getVaList',
  'isKnownUniquelyReferenced',
  'max',
  'min',
  'numericCast',
  'precondition',
  'preconditionFailure',
  'print',
  'readLine',
  'repeatElement',
  'sequence',
  'stride',
  'swap',
  'transcode',
  'type',
  'unsafeBitCast',
  'unsafeDowncast',
  'withExtendedLifetime',
  'withUnsafeBytes',
  'withUnsafeMutableBytes',
  'withUnsafeMutablePointer',
  'withUnsafePointer',
  'withVaList',
  'withoutActuallyEscaping',
  'zip',
]);

const SWIFT_CONSTANTS = new Set([
  'false',
  'nil',
  'true',
]);

const SWIFT_TYPES = new Set([
  'ASCII',
  'AcceptPolicy',
  'ActivationType',
  'AdditiveArithmetic',
  'AffineTransform',
  'AllCases',
  'Any',
  'AnyBidirectionalCollection',
  'AnyClass',
  'AnyCollection',
  'AnyHashable',
  'AnyIndex',
  'AnyIterator',
  'AnyKeyPath',
  'AnyObject',
  'AnyRandomAccessCollection',
  'AnySequence',
  'Array',
  'ArrayLiteralElement',
  'ArraySlice',
  'Behavior',
  'BinaryFloatingPoint',
  'BinaryInteger',
  'BiquadFunctions',
  'BlockOperation',
  'Body',
  'Bool',
  'BooleanLiteralType',
  'Bound',
  'Bundle',
  'ByteCountFormatter',
  'CBool',
  'CChar',
  'CChar16',
  'CChar32',
  'CDouble',
  'CFloat',
  'CGFloat',
  'CInt',
  'CLong',
  'CLongDouble',
  'CLongLong',
  'CShort',
  'CSignedChar',
  'CUnsignedChar',
  'CUnsignedInt',
  'CUnsignedLong',
  'CUnsignedLongLong',
  'CUnsignedShort',
  'CVaListPointer',
  'CVarArg',
  'CWideChar',
  'CachePolicy',
  'CachedURLResponse',
  'CalculationError',
  'Calendar',
  'CanonicalCombiningClass',
  'CaseIterable',
  'CenterType',
  'Change',
  'Character',
  'CharacterSet',
  'CheckingType',
  'Child',
  'Children',
  'CloseCode',
  'ClosedRange',
  'CocoaError',
  'Codable',
  'Code',
  'CodeUnit',
  'CodingKey',
  'CodingUserInfoKey',
  'Collection',
  'CollectionDifference',
  'CollectionOfOne',
  'CommandLine',
  'Comparable',
  'Comparator',
  'ComparisonResult',
  'CompletionHandler',
  'Component',
  'CompressionAlgorithm',
  'ContentKind',
  'Context',
  'ContiguousArray',
  'CountStyle',
  'CountableClosedRange',
  'CountablePartialRangeFrom',
  'CountableRange',
  'CustomDebugStringConvertible',
  'CustomLeafReflectable',
  'CustomNSError',
  'CustomPlaygroundDisplayConvertible',
  'CustomReflectable',
  'CustomStringConvertible',
  'DFTFunctions',
  'DTDKind',
  'Data',
  'DataDecodingStrategy',
  'DataEncodingStrategy',
  'DataTaskPublisher',
  'Date',
  'DateComponents',
  'DateComponentsFormatter',
  'DateDecodingStrategy',
  'DateEncodingStrategy',
  'DateFormatter',
  'DateInterval',
  'DateIntervalFormatter',
  'DateTimeStyle',
  'Deallocator',
  'Decimal',
  'Decodable',
  'Decoder',
  'DecodingError',
  'DecodingFailurePolicy',
  'DefaultIndices',
  'DefaultStringInterpolation',
  'DelayedRequestDisposition',
  'Dictionary',
  'DictionaryIndex',
  'DictionaryIterator',
  'DictionaryLiteral',
  'Dimension',
  'DirectoryEnumerator',
  'DisplayStyle',
  'Distance',
  'DistributedNotificationCenter',
  'DocumentAttributeKey',
  'DocumentReadingOptionKey',
  'DocumentType',
  'Double',
  'DropFirstSequence',
  'DropWhileSequence',
  'Element',
  'Elements',
  'EmptyCollection',
  'Encodable',
  'EncodedScalar',
  'Encoder',
  'Encoding',
  'EncodingError',
  'EnergyFormatter',
  'EnumeratedIterator',
  'EnumeratedSequence',
  'Equatable',
  'Error',
  'ErrorCode',
  'ErrorPointer',
  'Event',
  'Exponent',
  'ExpressionType',
  'ExtendedGraphemeClusterLiteralType',
  'ExtendedGraphemeClusterType',
  'ExternalEntityResolvingPolicy',
  'Failure',
  'FileAttributeKey',
  'FileAttributeType',
  'FileHandle',
  'FileManager',
  'FileManagerDelegate',
  'FileOperationKind',
  'FileProtectionType',
  'FileWrapper',
  'FixedWidthInteger',
  'FlattenCollection',
  'FlattenSequence',
  'Float',
  'Float32',
  'Float64',
  'Float80',
  'FloatLiteralType',
  'FloatingPoint',
  'FloatingPointClassification',
  'FloatingPointRoundingRule',
  'FloatingPointSign',
  'Formatter',
  'ForwardParser',
  'GeneralCategory',
  'HMCharacteristicPropertySupportsEvent',
  'HTTPCookie',
  'HTTPCookiePropertyKey',
  'HTTPCookieStorage',
  'HTTPCookieStringPolicy',
  'HTTPURLResponse',
  'Hashable',
  'Hasher',
  'Host',
  'ISO8601DateFormatter',
  'Identifier',
  'Index',
  'IndexDistance',
  'IndexPath',
  'IndexSet',
  'IndexingIterator',
  'Indices',
  'Input',
  'InputStream',
  'InsertionPosition',
  'Int',
  'Int16',
  'Int32',
  'Int64',
  'Int8',
  'IntegerLiteralType',
  'Iterator',
  'IteratorProtocol',
  'IteratorSequence',
  'JSONDecoder',
  'JSONEncoder',
  'JSONSerialization',
  'JoinedSequence',
  'Key',
  'KeyPath',
  'KeyValuePairs',
  'Keys',
  'Kind',
  'LanguageDirection',
  'LazyCollection',
  'LazyCollectionProtocol',
  'LazyDropWhileCollection',
  'LazyDropWhileSequence',
  'LazyFilterCollection',
  'LazyFilterSequence',
  'LazyMapCollection',
  'LazyMapSequence',
  'LazyPrefixWhileCollection',
  'LazyPrefixWhileSequence',
  'LazySequence',
  'LazySequenceProtocol',
  'LengthFormatter',
  'LoadHandler',
  'Locale',
  'LocalizedError',
  'LogicalType',
  'LosslessStringConvertible',
  'MachError',
  'Magnitude',
  'ManagedBuffer',
  'ManagedBufferPointer',
  'MaskStorage',
  'MassFormatter',
  'MatchingFlags',
  'MatchingPolicy',
  'Measurement',
  'MeasurementFormatter',
  'MemoryLayout',
  'Message',
  'MessagePort',
  'Mirror',
  'MirrorPath',
  'Mode',
  'Modifier',
  'NSAppleScript',
  'NSArchiver',
  'NSArray',
  'NSCache',
  'NSCacheDelegate',
  'NSCalendar',
  'NSCannotCreateScriptCommandError',
  'NSCharacterSet',
  'NSChineseCalendar',
  'NSClassDescription',
  'NSCloneCommand',
  'NSCloseCommand',
  'NSCloudSharingConflictError',
  'NSCloudSharingErrorMaximum',
  'NSCloudSharingErrorMinimum',
  'NSCloudSharingNetworkFailureError',
  'NSCloudSharingNoPermissionError',
  'NSCloudSharingOtherError',
  'NSCloudSharingQuotaExceededError',
  'NSCloudSharingTooManyParticipantsError',
  'NSCocoaErrorDomain',
  'NSCoder',
  'NSCoderErrorMaximum',
  'NSCoderErrorMinimum',
  'NSCoderInvalidValueError',
  'NSCoderReadCorruptError',
  'NSCoderValueNotFoundError',
  'NSCoding',
  'NSCollectionChangeType',
  'NSCollectorDisabledOption',
  'NSComparisonPredicate',
  'NSCompoundPredicate',
  'NSCompressionErrorMaximum',
  'NSCompressionErrorMinimum',
  'NSCompressionFailedError',
  'NSCondition',
  'NSConditionLock',
  'NSCopying',
  'NSCountCommand',
  'NSCountedSet',
  'NSCreateCommand',
  'NSData',
  'NSDataDetector',
  'NSDate',
  'NSDateComponentUndefined',
  'NSDateComponents',
  'NSDateInterval',
  'NSDebugDescriptionErrorKey',
  'NSDecimalMaxSize',
  'NSDecimalNoScale',
  'NSDecimalNumber',
  'NSDecimalNumberBehaviors',
  'NSDecimalNumberHandler',
  'NSDecompressionFailedError',
  'NSDeleteCommand',
  'NSDictionary',
  'NSDistributedLock',
  'NSEdgeInsets',
  'NSEdgeInsetsZero',
  'NSEnumerator',
  'NSError',
  'NSErrorDomain',
  'NSErrorFailingURLStringKey',
  'NSErrorPointer',
  'NSException',
  'NSExceptionName',
  'NSExecutableArchitectureMismatchError',
  'NSExecutableErrorMaximum',
  'NSExecutableErrorMinimum',
  'NSExecutableLinkError',
  'NSExecutableLoadError',
  'NSExecutableNotLoadableError',
  'NSExecutableRuntimeMismatchError',
  'NSExistsCommand',
  'NSExpression',
  'NSFastEnumeration',
  'NSFastEnumerationIterator',
  'NSFastEnumerationState',
  'NSFeatureUnsupportedError',
  'NSGetCommand',
  'NSGrammarCorrections',
  'NSGrammarRange',
  'NSGrammarUserDescription',
  'NSGregorianCalendar',
  'NSHPUXOperatingSystem',
  'NSHashEnumerator',
  'NSHashTable',
  'NSHebrewCalendar',
  'NSHelpAnchorErrorKey',
  'NSISO2022JPStringEncoding',
  'NSISO8601Calendar',
  'NSISOLatin1StringEncoding',
  'NSISOLatin2StringEncoding',
  'NSIndexPath',
  'NSIndexSet',
  'NSKeySpecifierEvaluationScriptError',
  'NSKeyValueChange',
  'NSKeyValueChangeKey',
  'NSKeyValueObservation',
  'NSKeyValueObservedChange',
  'NSKeyValueObservingCustomization',
  'NSKeyValueOperator',
  'NSKeyValueSetMutationKind',
  'NSKeyValueValidationError',
  'NSKeyedArchiveRootObjectKey',
  'NSKeyedArchiver',
  'NSKeyedArchiverDelegate',
  'NSKeyedUnarchiver',
  'NSKeyedUnarchiverDelegate',
  'NSLinguisticTag',
  'NSLinguisticTagScheme',
  'NSLinguisticTagger',
  'NSLinguisticTaggerUnit',
  'NSLoadedClasses',
  'NSLocale',
  'NSLock',
  'NSLocking',
  'NSLogicalTest',
  'NSMACHOperatingSystem',
  'NSMacOSRomanStringEncoding',
  'NSMachErrorDomain',
  'NSMachPort',
  'NSMachPortDelegate',
  'NSMapEnumerator',
  'NSMapTable',
  'NSMapTableCopyIn',
  'NSMapTableObjectPointerPersonality',
  'NSMapTableStrongMemory',
  'NSMapTableWeakMemory',
  'NSMeasurement',
  'NSMiddleSpecifier',
  'NSMoveCommand',
  'NSMutableArray',
  'NSMutableAttributedString',
  'NSMutableCharacterSet',
  'NSMutableCopying',
  'NSMutableData',
  'NSMutableDictionary',
  'NSMutableIndexSet',
  'NSMutableOrderedSet',
  'NSMutableSet',
  'NSMutableString',
  'NSMutableURLRequest',
  'NSNEXTSTEPStringEncoding',
  'NSNameSpecifier',
  'NSNoScriptError',
  'NSNoSpecifierError',
  'NSNotFound',
  'NSNotification',
  'NSNull',
  'NSNumber',
  'NSObject',
  'NSObjectProtocol',
  'NSOrderedSet',
  'NSOrthography',
  'NSPoint',
  'NSPointerArray',
  'NSPointerFunctions',
  'NSPositionalSpecifier',
  'NSPredicate',
  'NSProxy',
  'NSPurgeableData',
  'NSQuitCommand',
  'NSRandomSpecifier',
  'NSRange',
  'NSRect',
  'NSRecursiveLock',
  'NSRegularExpression',
  'NSRelativeSpecifier',
  'NSScannedOption',
  'NSSecureCoding',
  'NSSet',
  'NSSetCommand',
  'NSSize',
  'NSSortDescriptor',
  'NSSpecifierTest',
  'NSSpellServer',
  'NSSpellServerDelegate',
  'NSStreamSOCKSErrorDomain',
  'NSStreamSocketSSLErrorDomain',
  'NSString',
  'NSTimeZone',
  'NSURL',
  'NSURLRequest',
  'NSUUID',
  'NSUnarchiver',
  'NSValue',
  'NSWhoseSpecifier',
  'Name',
  'NameStyle',
  'NetService',
  'NetServiceBrowser',
  'NetServiceBrowserDelegate',
  'NetServiceDelegate',
  'NetworkServiceType',
  'NetworkUnavailableReason',
  'Never',
  'Notification',
  'NotificationCenter',
  'NotificationCoalescing',
  'NotificationQueue',
  'NumberFormatter',
  'NumberRepresentation',
  'Numeric',
  'NumericType',
  'ObjectIdentifier',
  'ObservableObject',
  'OpaquePointer',
  'OperatingSystemVersion',
  'Operation',
  'OperationQueue',
  'Operator',
  'Optional',
  'Output',
  'OutputFormatting',
  'OutputStream',
  'POSIXError',
  'PadPosition',
  'ParseResult',
  'Parser',
  'PartialKeyPath',
  'PartialRangeFrom',
  'PartialRangeThrough',
  'PartialRangeUpTo',
  'Persistence',
  'Pipe',
  'PlaygroundQuickLook',
  'Pointee',
  'Port',
  'PortDelegate',
  'PortMessage',
  'PostingStyle',
  'PrefixSequence',
  'Process',
  'ProcessInfo',
  'Progress',
  'ProgressKind',
  'ProgressReporting',
  'ProgressUserInfoKey',
  'Properties',
  'PropertyKey',
  'Published',
  'Publisher',
  'PublishingHandler',
  'QualityOfService',
  'QueuePriority',
  'RandomAccessCollection',
  'RandomNumberGenerator',
  'Range',
  'RangeExpression',
  'RangeReplaceableCollection',
  'RangeView',
  'RawExponent',
  'RawRepresentable',
  'RawSignificand',
  'RawValue',
  'RecoverableError',
  'ReferenceConvertible',
  'ReferenceType',
  'ReferenceWritableKeyPath',
  'Regions',
  'RelativePosition',
  'Repeated',
  'RepeatedTimePolicy',
  'ResourceFetchType',
  'ResponseDisposition',
  'Result',
  'ReverseParser',
  'ReversedCollection',
  'RoundingMode',
  'RunLoop',
  'SIMD',
  'SIMD16',
  'SIMD16Storage',
  'SIMD2',
  'SIMD2Storage',
  'SIMD3',
  'SIMD32',
  'SIMD32Storage',
  'SIMD4',
  'SIMD4Storage',
  'SIMD64',
  'SIMD64Storage',
  'SIMD8',
  'SIMD8Storage',
  'SIMDMask',
  'SIMDMaskScalar',
  'SIMDScalar',
  'SIMDStorage',
  'Scalar',
  'Scanner',
  'SchedulerTimeType',
  'SearchDirection',
  'SearchPathDirectory',
  'SearchPathDomainMask',
  'Sequence',
  'Set',
  'SetAlgebra',
  'SetIndex',
  'SetIterator',
  'SignedInteger',
  'SignedNumeric',
  'Slice',
  'SocketNativeHandle',
  'SocketPort',
  'SpellingState',
  'State',
  'StaticString',
  'Status',
  'StoragePolicy',
  'Stream',
  'StreamDelegate',
  'Stride',
  'StrideThrough',
  'StrideThroughIterator',
  'StrideTo',
  'StrideToIterator',
  'Strideable',
  'String',
  'StringInterpolation',
  'StringInterpolationProtocol',
  'StringLiteralType',
  'StringProtocol',
  'StringTransform',
  'Style',
  'SubSequence',
  'SubelementIdentifier',
  'Substring',
  'SuspensionBehavior',
  'SuspensionID',
  'SystemRandomNumberGenerator',
  'TerminationReason',
  'TestComparisonOperation',
  'TextEffectStyle',
  'TextLayoutSectionKey',
  'TextOutputStream',
  'TextOutputStreamable',
  'ThermalState',
  'Thread',
  'TimeInterval',
  'TimeZone',
  'Timer',
  'TimerPublisher',
  'Tuple',
  'UInt',
  'UInt16',
  'UInt32',
  'UInt64',
  'UInt8',
  'URL',
  'URLCache',
  'URLComponents',
  'URLCredential',
  'URLCredentialStorage',
  'URLError',
  'URLFileProtection',
  'URLFileResourceType',
  'URLProtectionSpace',
  'URLProtocol',
  'URLProtocolClient',
  'URLQueryItem',
  'URLRelationship',
  'URLRequest',
  'URLResourceKey',
  'URLResourceValues',
  'URLResponse',
  'URLSession',
  'UTF16',
  'UTF16View',
  'UTF32',
  'UTF8',
  'UTF8View',
  'UUID',
  'UnboundedRange',
  'UndoManager',
  'UnfoldFirstSequence',
  'UnfoldSequence',
  'Unicode',
  'UnicodeCodec',
  'UnicodeDecodingResult',
  'UnicodeScalar',
  'UnicodeScalarIndex',
  'UnicodeScalarLiteralType',
  'UnicodeScalarType',
  'UnicodeScalarView',
  'Unit',
  'UnitAcceleration',
  'UnitAngle',
  'UnitArea',
  'UnitConcentrationMass',
  'UnitConverter',
  'UnitConverterLinear',
  'UnitDispersion',
  'UnitDuration',
  'UnitEnergy',
  'UnitFrequency',
  'UnitFuelEfficiency',
  'UnitIlluminance',
  'UnitLength',
  'UnitMass',
  'UnitPower',
  'UnitPressure',
  'UnitSpeed',
  'UnitStyle',
  'UnitTemperature',
  'UnitVolume',
  'Units',
  'UnitsStyle',
  'Unmanaged',
  'UnpublishingHandler',
  'UnsafeBufferPointer',
  'UnsafeMutableBufferPointer',
  'UnsafeMutablePointer',
  'UnsafeMutableRawBufferPointer',
  'UnsafeMutableRawPointer',
  'UnsafePointer',
  'UnsafeRawBufferPointer',
  'UnsafeRawPointer',
  'UnsignedInteger',
  'UserDefaults',
  'UserInfoKey',
  'Value',
  'ValueTransformer',
  'Values',
  'Version',
  'Void',
  'Words',
  'XMLDTD',
  'XMLDTDNode',
  'XMLDocument',
  'XMLElement',
  'XMLNode',
  'XMLParser',
  'XMLParserDelegate',
  'Zip2Sequence',
  'swift',
  'unichar',
]);

class HighlightSwift extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static SLASH = 2;
  static SLASH_SLASH = 3;
  static SLASH_STAR = 4;
  static SLASH_SLASH_BACKSLASH = 5;
  static SLASH_STAR_STAR = 6;
  static HASH = 7;
  static DQUOTE = 8;
  static DQUOTESTR = 9;
  static DQUOTESTR_BACKSLASH = 10;
  static DQUOTESTR_END = 11;
  static DQUOTE2 = 12;
  static DQUOTE3 = 13;
  static DQUOTE3_BACKSLASH = 14;
  static DQUOTE31 = 15;
  static DQUOTE32 = 16;
  static DQUOTE3_END = 17;
  static REGEX = 18;
  static REGEX_END = 19;
  static REGEX_BACKSLASH = 20;

  static EXPECT_VALUE = 0;
  static EXPECT_OPERATOR = 1;

  constructor(delegate) {
    super(delegate);
    this.hash1 = 0;
    this.hash2 = 0;
    this.word = '';
    this.nest = [];
    this.hash = [];
    this.expect = HighlightSwift.EXPECT_VALUE;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightSwift.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightSwift.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightSwift.SLASH;
        } else if (c == '"') {
          this.state = HighlightSwift.DQUOTE;
          this.push("span", "string");
          this.append('"');
          this.hash1 = 0;
          this.expect = HighlightSwift.EXPECT_OPERATOR;
        } else if (c == '#') {
          this.state = HighlightSwift.HASH;
          this.hash1 = 1;
          this.expect = HighlightSwift.EXPECT_OPERATOR;
        } else if (c == '(' && this.nest.length) {
          this.append('(');
          this.nest.push(HighlightSwift.NORMAL);
          this.hash.push(0);
          this.expect = HighlightSwift.EXPECT_VALUE;
        } else if (c == ')' && this.nest.length) {
          this.expect = HighlightSwift.EXPECT_OPERATOR;
          this.state = this.nest.pop();
          this.hash1 = this.hash.pop();
          if (this.state != HighlightSwift.NORMAL)
            this.push("span", "string");
          this.append(')');
        } else if (c == ')' || c == ']' || isdigit(c) || c == '.') {
          this.expect = HighlightSwift.EXPECT_OPERATOR;
          this.append(c);
        } else if (ispunct(c) || c == '\n') {
          this.expect = HighlightSwift.EXPECT_VALUE;
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightSwift.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (SWIFT_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (SWIFT_TYPES.has(this.word)) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (SWIFT_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (SWIFT_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightSwift.NORMAL);
        }
        break;

      case HighlightSwift.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightSwift.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightSwift.SLASH_STAR;
        } else if (expect_ == EXPECT_VALUE) {
          this.push("span", "string");
          this.append('/');
          this.hash1 = 0;
          this.epsilon(HighlightSwift.REGEX);
        } else {
          this.append('/');
          this.epsilon(HighlightSwift.NORMAL);
        }
        break;

      case HighlightSwift.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightSwift.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightSwift.SLASH_SLASH_BACKSLASH;
        }
        break;

      case HighlightSwift.SLASH_SLASH_BACKSLASH:
        this.append(c);
        this.state = HighlightSwift.SLASH_SLASH;
        break;

      case HighlightSwift.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightSwift.SLASH_STAR_STAR;
        break;

      case HighlightSwift.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightSwift.NORMAL;
        } else if (c != '*') {
          this.state = HighlightSwift.SLASH_STAR;
        }
        break;

      case HighlightSwift.HASH:
        if (c == '#') {
          ++this.hash1;
        } else if (c == '"') {
          this.push("span", "string");
          for (let i = 0; i < this.hash1; ++i)
            this.append('#');
          this.append('"');
          this.state = HighlightSwift.DQUOTE;
        } else if (c == '/') {
          this.push("span", "string");
          for (let i = 0; i < this.hash1; ++i)
            this.append('#');
          this.append('/');
          this.state = HighlightSwift.REGEX;
        } else {
          for (let i = 0; i < this.hash1; ++i)
            this.append('#');
          this.epsilon(HighlightSwift.NORMAL);
        }
        break;

      case HighlightSwift.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightSwift.DQUOTE2;
          this.hash2 = 0;
        } else if (c == '\\') {
          this.state = HighlightSwift.DQUOTESTR_BACKSLASH;
          this.hash2 = 0;
        } else {
          this.state = HighlightSwift.DQUOTESTR;
        }
        break;

      case HighlightSwift.DQUOTESTR:
        this.append(c);
        if (c == '"') {
          this.state = HighlightSwift.DQUOTESTR_END;
          this.hash2 = 0;
        } else if (c == '\\') {
          this.state = HighlightSwift.DQUOTESTR_BACKSLASH;
          this.hash2 = 0;
        }
        break;

      case HighlightSwift.DQUOTESTR_END:
        if (this.hash2 == this.hash1) {
          this.pop();
          this.epsilon(HighlightSwift.NORMAL);
        } else if (c == '#') {
          this.append('#');
          ++this.hash2;
        } else {
          this.epsilon(HighlightSwift.DQUOTESTR);
        }
        break;

      case HighlightSwift.DQUOTESTR_BACKSLASH:
        if (c == '#' && this.hash2 < this.hash1) {
          this.append('#');
          ++this.hash2;
        } else if (c == '(' && this.hash2 == this.hash1) {
          this.append('(');
          this.pop();
          this.nest.push(HighlightSwift.DQUOTESTR);
          this.hash.push(this.hash1);
          this.state = HighlightSwift.NORMAL;
        } else {
          this.epsilon(HighlightSwift.DQUOTESTR);
        }
        break;

      case HighlightSwift.DQUOTE2:
        if (c == '"') {
          this.append('"');
          this.state = HighlightSwift.DQUOTE3;
        } else if (c == '#' && this.hash2 < this.hash1) {
          this.append('#');
          ++this.hash2;
        } else if (this.hash2 == this.hash1) {
          this.pop();
          this.epsilon(HighlightSwift.NORMAL);
        } else {
          this.epsilon(HighlightSwift.DQUOTESTR);
        }
        break;

      case HighlightSwift.DQUOTE3:
        this.append(c);
        if (c == '"') {
          this.state = HighlightSwift.DQUOTE31;
        } else if (c == '\\') {
          this.state = HighlightSwift.DQUOTE3_BACKSLASH;
          this.hash2 = 0;
        }
        break;

      case HighlightSwift.DQUOTE31:
        if (c == '"') {
          this.append('"');
          this.state = HighlightSwift.DQUOTE32;
        } else {
          this.epsilon(HighlightSwift.DQUOTE3);
        }
        break;

      case HighlightSwift.DQUOTE32:
        if (c == '"') {
          this.append('"');
          this.state = HighlightSwift.DQUOTESTR_END;
          this.hash2 = 0;
        } else {
          this.epsilon(HighlightSwift.DQUOTE3);
        }
        break;

      case HighlightSwift.DQUOTE3_BACKSLASH:
        if (c == '#' && this.hash2 < this.hash1) {
          this.append('#');
          ++this.hash2;
        } else if (c == '(' && this.hash2 == this.hash1) {
          this.append('(');
          this.pop();
          this.nest.push(HighlightSwift.DQUOTE3);
          this.hash.push(this.hash1);
          this.state = HighlightSwift.NORMAL;
        } else {
          this.epsilon(HighlightSwift.DQUOTE3);
        }
        break;

      case HighlightSwift.DQUOTE3_END:
        if (this.hash2 == this.hash1) {
          this.pop();
          this.epsilon(HighlightSwift.NORMAL);
        } else if (c == '#') {
          this.append('#');
          ++this.hash2;
        } else {
          this.epsilon(HighlightSwift.DQUOTE3);
        }
        break;

      case HighlightSwift.REGEX:
        this.append(c);
        if (c == '/') {
          this.state = HighlightSwift.REGEX_END;
          this.hash2 = 0;
        } else if (c == '\\') {
          this.state = HighlightSwift.REGEX_BACKSLASH;
          this.hash2 = 0;
        }
        break;

      case HighlightSwift.REGEX_END:
        if (this.hash2 == this.hash1) {
          this.pop();
          this.epsilon(HighlightSwift.NORMAL);
        } else if (c == '#') {
          this.append('#');
          ++this.hash2;
        } else {
          this.epsilon(HighlightSwift.REGEX);
        }
        break;

      case HighlightSwift.REGEX_BACKSLASH:
        this.append(c);
        this.state = HighlightSwift.REGEX;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightSwift.WORD:
      if (SWIFT_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (SWIFT_TYPES.has(this.word)) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (SWIFT_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (SWIFT_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightSwift.SLASH:
      this.append('/');
      break;
    case HighlightSwift.HASH:
      for (let i = 0; i < this.hash1; ++i)
        this.append('#');
      break;
    case HighlightSwift.SLASH_SLASH:
    case HighlightSwift.SLASH_SLASH_BACKSLASH:
    case HighlightSwift.SLASH_STAR:
    case HighlightSwift.SLASH_STAR_STAR:
    case HighlightSwift.DQUOTE:
    case HighlightSwift.DQUOTESTR:
    case HighlightSwift.DQUOTESTR_BACKSLASH:
    case HighlightSwift.DQUOTESTR_END:
    case HighlightSwift.DQUOTE2:
    case HighlightSwift.DQUOTE3:
    case HighlightSwift.DQUOTE3_BACKSLASH:
    case HighlightSwift.DQUOTE31:
    case HighlightSwift.DQUOTE32:
    case HighlightSwift.DQUOTE3_END:
    case HighlightSwift.REGEX:
    case HighlightSwift.REGEX_END:
    case HighlightSwift.REGEX_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightSwift.NORMAL;
    this.nest = [];
    this.hash = [];
    this.delegate.flush();
    this.delta = 1;
    this.expect = HighlightSwift.EXPECT_VALUE;
  }
}

Highlighter.REGISTRY['swift'] = HighlightSwift;
