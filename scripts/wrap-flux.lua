-------------------------------------------------------------------------------
-- Wrap flux data for integration in a C binary
-- Author: Valentin Niess
-------------------------------------------------------------------------------
local ffi = require('ffi')

ffi.cdef([[
size_t fread(void * ptr, size_t size, size_t nmemb, void * stream);
]])

local path = arg[1] or 'data/flux-mceq-sybill23c-polygonato-usstd.table'
local filename = arg[2] or 'flux_mceq.c'

local file = io.open(path)
if not file then
    error('could not open '..path)
end

local shape = ffi.new('int64_t [3]')
local range = ffi.new('double [6]')
ffi.C.fread(shape, 8, 3, file)
ffi.C.fread(range, 8, 6, file)
local size = tonumber(2 * shape[0] * shape[1] * shape[2])
local data = ffi.new('float [?]', size)
ffi.C.fread(data, 4, size, file)
file:close()

file = io.open(filename, 'w')
file:write(string.format([[
static struct {
        struct pumas_flux_tabulation base;
        float data[%d];
} pumas_flux_mceq = {{%d, %d, %d, %.9E, %.9E, %.9E, %.9E, %.9E, %.9E},
    {%.9E]], size, tonumber(shape[0]), tonumber(shape[1]), tonumber(shape[2]),
    range[0], range[1], range[2], range[3], range[4], range[5], data[0]))
for i = 1, size - 1 do
    file:write(string.format(', %.9E', data[i]))
end
file:write([[}};
]])
file:close()
